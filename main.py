import argparse
import os
import sys
import json
from datetime import datetime
import time
import cProfile
import pstats
import io
from data_loader import load_musique_dataset, get_example_by_id, load_hotpotqa_dataset, load_dataset
from deepseek_api import test_api, set_max_concurrent_requests
from entity_extractor import EntityExtractor
from graph_visualizer import visualize_entity_document_graph, visualize_entity_relationship_graph
from difflib import SequenceMatcher
import pandas as pd
import concurrent.futures
import traceback
from tqdm import tqdm
from queue import Queue
import threading
from rate_limiter import get_rate_limiter
import networkx as nx
import matplotlib.pyplot as plt
from evaluation import AnswerEvaluator
from stratified_sampler import stratified_sample, get_hop_distribution

# Profiling decorator
def profile_func(func):
    def wrapper(*args, **kwargs):
        profiler = cProfile.Profile()
        profiler.enable()
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        profiler.disable()
        
        # Print execution time
        print(f"\n{'='*50}")
        print(f"PROFILING: {func.__name__}")
        print(f"Total execution time: {end_time - start_time:.2f} seconds")
        
        # Get profiling stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
        ps.print_stats(20)  # Print top 20 functions by cumulative time
        print(s.getvalue())
        
        return result
    return wrapper

def process_single_question(example_id, df, args, output_dir, shared_extractor=None):
    """
    Process a single question and save results
    
    Args:
        example_id: ID of the example to process
        df: DataFrame containing the dataset
        args: Command line arguments
        output_dir: Directory to save results
        shared_extractor: Optional shared EntityExtractor instance
        
    Returns:
        Dictionary with experiment results
    """
    try:
        print(f"\nProcessing example with ID: {example_id}")
        
        # Create a directory for this example
        example_dir = os.path.join(output_dir, f"example_{example_id}")
        os.makedirs(example_dir, exist_ok=True)
        
        # Get the example
        example = df.iloc[int(example_id)]
        
        # Save question info
        question_info = {
            'question': example['question'],
            'answer': example['answer'],
            'dataset_type': args.dataset_type
        }
        
        # Add dataset-specific fields
        if args.dataset_type.lower() == 'hotpotqa':
            question_info['type'] = example.get('type', '')
            question_info['level'] = example.get('level', '')
        
        with open(os.path.join(example_dir, "question_info.json"), "w") as f:
            json.dump(question_info, f, indent=2)
        
        print(f"\nQuestion: {example['question']}")
        print(f"Answer: {example['answer']}")
        
        # Determine which experiments to run
        experiments = ['standard', 'fuzzy_matching', 'llm_merging', 'sequential_context', 'all_docs_baseline', 'supporting_docs_baseline'] if args.experiment == 'all' else [args.experiment]
        
        # Store results for comparison
        experiment_results = []
        timing_data = {}
        
        # Use the shared extractor if provided, otherwise create a new one
        extractor = shared_extractor or EntityExtractor(
            verbose=args.verbose, 
            model_name=args.model,
            use_cache=not args.disable_cache,
            dataset_type=args.dataset_type  # Pass dataset type to EntityExtractor
        )
        
        # Check if we have cached data for this example
        cached_data_available = extractor.check_example_cache(example_id, example)
        if cached_data_available and not args.disable_cache:
            print(f"Found cached data for example {example_id}")
        
        # Extract entities from question and paragraphs once for all experiments
        print("\nExtracting entities from question and paragraphs (shared across experiments)...")
        start_time = time.time()
        
        # Initialize a set to store all known entities across experiments
        all_known_entities = set()

        # For sequential_context experiment, we need to extract entities first
        if 'sequential_context' in experiments:
            print("Pre-extracting entities using sequential context approach...")
            # Start with an empty set of known entities
            known_entities = set()
            
            # Process paragraphs sequentially to build up entity knowledge
            for i, paragraph in enumerate(tqdm(example['paragraphs'], desc="Building entity knowledge")):
                # Create a document ID
                doc_id = f"example_{example_id}_doc_{i}"
                
                # Extract entities with context
                entities = extractor._extract_entities_with_context(
                    paragraph['paragraph_text'], 
                    list(known_entities),
                    doc_id=doc_id
                )
                
                # Update known entities
                known_entities.update(entities)
                
                # Print entities found in this paragraph
                if args.verbose:
                    print(f"Paragraph {i} ({paragraph['title']}): {entities}")
            
            # Store all known entities for other experiments
            all_known_entities = known_entities
            print(f"Found {len(all_known_entities)} entities using sequential context")

        # Extract question entities using known entities from sequential context if available
        question_entities = extractor.extract_entities_from_question(
            example['question'], 
            list(all_known_entities) if all_known_entities else None,
            question_id=example_id
        )
        print(f"Entities in question: {question_entities}")
        
        # Pre-extract paragraph entities for all experiments
        paragraph_entities = {}
        if any(exp in ['standard', 'fuzzy_matching', 'llm_merging'] for exp in experiments):
            print(f"Pre-extracting entities from {len(example['paragraphs'])} paragraphs...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
                # Create a list to store futures
                futures = []
                
                # Submit tasks for each paragraph
                for i, paragraph in enumerate(example['paragraphs']):
                    # Create a document ID
                    doc_id = f"example_{example_id}_doc_{i}"
                    
                    futures.append(
                        executor.submit(
                            extractor.extract_entities_from_paragraph,
                            paragraph['paragraph_text'],
                            doc_id=doc_id
                        )
                    )
                
                # Process results as they complete
                for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), 
                                              total=len(futures),
                                              desc="Pre-extracting entities")):
                    try:
                        # Get the paragraph and its entities
                        paragraph = example['paragraphs'][i]
                        entities = future.result()
                        paragraph_entities[i] = entities
                        
                    except Exception as e:
                        print(f"Error pre-extracting paragraph {i}: {e}")
        
        entity_extraction_time = time.time() - start_time
        print(f"Entity extraction completed in {entity_extraction_time:.2f} seconds")
        
        # Run each experiment
        for experiment in experiments:
            print(f"\nRunning experiment: {experiment}")
            exp_dir = os.path.join(example_dir, experiment)
            os.makedirs(exp_dir, exist_ok=True)
            
            # Create a new graph for this experiment
            G = nx.Graph()
            
            # Add question entities to the graph
            for entity in question_entities:
                G.add_node(entity, type='entity')
            
            # Apply the appropriate experiment
            start_time = time.time()
            
            if experiment == 'standard':
                G, mapped_question_entities = extractor.apply_standard_experiment(G, example, question_entities)
            elif experiment == 'fuzzy_matching':
                G, mapped_question_entities = extractor.apply_fuzzy_matching_experiment(G, example, question_entities)
            elif experiment == 'llm_merging':
                G, mapped_question_entities = extractor.apply_llm_merging_experiment(G, example, question_entities)
            elif experiment == 'sequential_context':
                G, mapped_question_entities = extractor.apply_sequential_context_experiment(G, example, question_entities, example_id=example_id)
            elif experiment == 'all_docs_baseline':
                # For the baseline, we'll create a minimal graph structure just to maintain compatibility
                # with the rest of the pipeline, but we won't do any entity extraction
                for i, paragraph in enumerate(example['paragraphs']):
                    doc_id = f"doc_{i}"
                    G.add_node(doc_id, 
                              type='document', 
                              title=paragraph['title'], 
                              text=paragraph['paragraph_text'],
                              is_supporting=paragraph.get('is_supporting', False))
                
                # No entity extraction or relationship building
                mapped_question_entities = question_entities
            elif experiment == 'supporting_docs_baseline':
                # For the supporting_docs_baseline experiment, we'll create a minimal graph structure just to maintain compatibility
                # with the rest of the pipeline, but we won't do any entity extraction
                for i, paragraph in enumerate(example['paragraphs']):
                    doc_id = f"doc_{i}"
                    G.add_node(doc_id, 
                              type='document', 
                              title=paragraph['title'], 
                              text=paragraph['paragraph_text'],
                              is_supporting=paragraph.get('is_supporting', False))
                
                # No entity extraction or relationship building
                mapped_question_entities = question_entities
            else:
                print(f"Unknown experiment: {experiment}")
                continue
            
            # Track timing for each major step
            timing = {}
            timing['graph_creation'] = time.time() - start_time
            print(f"Graph creation completed in {timing['graph_creation']:.2f} seconds")
            
            # After creating the bipartite graph and before finding reachable documents
            # Save the bipartite graph as JSON for the dashboard
            bipartite_graph_data = {
                "nodes": [
                    {
                        "id": str(n),
                        "type": G.nodes[n].get('type', 'unknown'),
                        "title": G.nodes[n].get('title', ''),
                        "is_supporting": G.nodes[n].get('is_supporting', False)
                    } 
                    for n in G.nodes()
                ],
                "edges": [
                    {
                        "source": str(u),
                        "target": str(v)
                    } 
                    for u, v in G.edges()
                ]
            }

            with open(os.path.join(exp_dir, "bipartite_graph.json"), "w") as f:
                json.dump(bipartite_graph_data, f, indent=2)

            # Save document classification results instead of pickle
            doc_classification = {
                "true_positives": [],  # Supporting docs that are reachable
                "false_negatives": [], # Supporting docs that are not reachable
                "false_positives": [], # Non-supporting docs that are reachable
                "true_negatives": []   # Non-supporting docs that are not reachable
            }

            # We'll populate this after finding reachable documents
            # Remove the pickle saving line
            # nx.write_gpickle(G, os.path.join(exp_dir, "bipartite_graph.gpickle"))

            # Find reachable documents
            print(f"\nFinding documents reachable within {args.max_hops} hops...")
            step_start = time.time()
            if experiment == 'all_docs_baseline':
                # For the baseline, all documents are "reachable"
                reachable_docs = [n for n in G.nodes() if G.nodes[n].get('type') == 'document']
                print(f"Using all {len(reachable_docs)} documents for baseline experiment")
            elif experiment == 'supporting_docs_baseline':
                # For the supporting_docs_baseline experiment, all documents are "reachable"
                # For the supporting_docs_baseline experiment, only supporting documents are "reachable"
                reachable_docs = [n for n in G.nodes() if G.nodes[n].get('type') == 'document' and G.nodes[n].get('is_supporting', False)]
                print(f"Using {len(reachable_docs)} supporting documents for supporting_docs_baseline experiment")
            else:
                reachable_docs = find_reachable_documents(G, question_entities, max_hops=args.max_hops)
            timing['document_retrieval'] = time.time() - step_start
            print(f"Document retrieval completed in {timing['document_retrieval']:.2f} seconds")
            
            # Count supporting documents
            supporting_docs = [doc_id for doc_id in reachable_docs if G.nodes[doc_id].get('is_supporting', False)]
            
            print(f"\nFound {len(reachable_docs)} reachable documents, {len(supporting_docs)} of which are supporting.")
            
            # Save reachable documents info
            with open(os.path.join(exp_dir, "reachable_docs.json"), "w") as f:
                json.dump({
                    "reachable_docs": list(reachable_docs),
                    "supporting_docs": supporting_docs
                }, f, indent=2)
            
            # Create entity relationship graph based on reachable documents
            print("\nCreating entity relationship graph...")
            step_start = time.time()
            if experiment == 'all_docs_baseline':
                # For the baseline, we create an empty entity graph (no relationships)
                entity_graph = nx.Graph()
            elif experiment == 'supporting_docs_baseline':
                # For the supporting_docs_baseline experiment, we create an empty entity graph (no relationships)
                entity_graph = nx.Graph()
            else:
                entity_graph = extractor.create_entity_relationship_graph(G, reachable_docs)
            timing['relationship_extraction'] = time.time() - step_start
            print(f"Relationship extraction completed in {timing['relationship_extraction']:.2f} seconds")
            
            print(f"Entity relationship graph has {entity_graph.number_of_nodes()} entities and {entity_graph.number_of_edges()} relationships")
            
            # Save the entity graph as JSON for the dashboard
            entity_graph_data = {
                "nodes": [{"id": str(n)} for n in entity_graph.nodes()],
                "edges": [
                    {
                        "source": str(u), 
                        "target": str(v), 
                        "relation": entity_graph.edges[u, v].get('relation', '')
                    } 
                    for u, v in entity_graph.edges()
                ]
            }

            with open(os.path.join(exp_dir, "entity_graph.json"), "w") as f:
                json.dump(entity_graph_data, f, indent=2)
            
            # Visualize the graphs and save them
            if not args.skip_visualization:
                print("\nVisualizing entity-document graph...")
                start_time = time.time()
                
                # Use non-GUI mode when running in parallel
                visualize_entity_document_graph(
                    G, 
                    question_entities=question_entities, 
                    reachable_docs=reachable_docs,
                    save_path=os.path.join(exp_dir, "entity_document_graph.png"),
                    non_interactive=shared_extractor is not None
                )
                
                bipartite_viz_time = time.time() - start_time
                print(f"Bipartite graph visualization completed in {bipartite_viz_time:.2f} seconds")
                
                print("\nVisualizing entity relationship graph...")
                start_time = time.time()
                
                visualize_entity_relationship_graph(
                    entity_graph, 
                    question_entities=question_entities,
                    save_path=os.path.join(exp_dir, "entity_relationship_graph.png"),
                    non_interactive=shared_extractor is not None
                )
                
                relationship_viz_time = time.time() - start_time
                print(f"Relationship graph visualization completed in {relationship_viz_time:.2f} seconds")
            
            # Generate answer based on entity relationships and documents
            print("\nGenerating answer based on entity relationships and documents...")
            step_start = time.time()
            if experiment == 'all_docs_baseline':
                # For the baseline, generate answer using all documents directly
                generated_answer = extractor.generate_answer_baseline(
                    example['question'],
                    reachable_docs,
                    G
                )
            elif experiment == 'supporting_docs_baseline':
                # For the supporting_docs_baseline experiment, generate answer using all documents directly
                generated_answer = extractor.generate_answer_baseline(
                    example['question'],
                    reachable_docs,
                    G
                )
            else:
                generated_answer = extractor.generate_answer(
                    example['question'], 
                    entity_graph, 
                    question_entities, 
                    reachable_docs,
                    G
                )
            timing['answer_generation'] = time.time() - step_start
            print(f"Answer generation completed in {timing['answer_generation']:.2f} seconds")
            
            # Calculate simple similarity between generated and expected answers
            def similarity(a, b):
                return SequenceMatcher(None, a, b).ratio()
            
            sim_score = similarity(generated_answer.lower(), example['answer'].lower())
            print(f"\nSimilarity score: {sim_score:.2f}")
            
            # Calculate total execution time
            execution_time = time.time() - start_time
            timing['total'] = execution_time
            
            # Store timing data
            timing_data[experiment] = timing
            
            # Create an evaluator
            evaluator = AnswerEvaluator()

            # Evaluate the answer
            evaluation_results = evaluator.evaluate(generated_answer, example["answer"])

            # Add evaluation results to the experiment results
            results = {
                "generated_answer": generated_answer,
                "reference_answer": example["answer"],
                "similarity_score": sim_score,
                "exact_match": evaluation_results["exact_match"],
                "partial_match": evaluation_results.get("partial_match", False),
                "f1_score": evaluation_results.get("f1_score", 0.0),
                "precision": evaluation_results.get("precision", 0.0),
                "recall": evaluation_results.get("recall", 0.0),
                "extracted_answer": evaluation_results["extracted_prediction"],
                "normalized_prediction": evaluation_results["normalized_prediction"],
                "normalized_reference": evaluation_results["normalized_reference"],
                "timing": timing_data,
                "execution_time": execution_time
            }
            
            # Save answer and metrics
            with open(os.path.join(exp_dir, "results.json"), "w") as f:
                json.dump(results, f, indent=2)
            
            # Update the experiment_results to include the new metrics
            experiment_results.append({
                'experiment': experiment,
                'reachable_docs': len(reachable_docs),
                'supporting_docs': len(supporting_docs),
                'entity_count': entity_graph.number_of_nodes(),
                'relationship_count': entity_graph.number_of_edges(),
                'answer': generated_answer,
                'similarity': sim_score,
                'exact_match': evaluation_results["exact_match"],
                'partial_match': evaluation_results.get("partial_match", False),
                'f1_score': evaluation_results.get("f1_score", 0.0),
                'precision': evaluation_results.get("precision", 0.0),
                'recall': evaluation_results.get("recall", 0.0),
                'execution_time': execution_time
            })
            
            # Then after finding reachable documents, add:
            # Categorize documents for metrics
            for node in G.nodes():
                if G.nodes[node].get('type') == 'document':
                    is_supporting = G.nodes[node].get('is_supporting', False)
                    is_reachable = node in reachable_docs
                    
                    if is_supporting and is_reachable:
                        doc_classification["true_positives"].append(node)
                    elif is_supporting and not is_reachable:
                        doc_classification["false_negatives"].append(node)
                    elif not is_supporting and is_reachable:
                        doc_classification["false_positives"].append(node)
                    else:  # not is_supporting and not is_reachable
                        doc_classification["true_negatives"].append(node)

            # Save document classification results
            with open(os.path.join(exp_dir, "doc_classification.json"), "w") as f:
                json.dump(doc_classification, f, indent=2)

            # After creating the bipartite graph and before generating the answer
            # Add this code to save the relationship graph:

            # Create and save the entity relationship graph
            entity_graph = nx.Graph()
            for node in G.nodes():
                if G.nodes[node].get('type') == 'entity':
                    entity_graph.add_node(node, type='entity')

            # Add edges between entities based on relationships
            for u, v, data in G.edges(data=True):
                if G.nodes[u].get('type') == 'entity' and G.nodes[v].get('type') == 'entity':
                    entity_graph.add_edge(u, v, relation=data.get('relation', ''), source_doc=data.get('source_doc', ''))

            # Save the relationship graph
            relationship_graph_path = os.path.join(exp_dir, "relationship_graph.json")
            nx.node_link_data(entity_graph)
            with open(relationship_graph_path, 'w') as f:
                json.dump(nx.node_link_data(entity_graph), f, indent=2)
        
        # If multiple experiments were run, save comparison
        if len(experiment_results) > 1:
            # Create a DataFrame for better visualization
            results_df = pd.DataFrame(experiment_results)
            
            # Save comparison to CSV
            results_df.to_csv(os.path.join(example_dir, "experiment_comparison.csv"), index=False)
            
            # Determine the best experiment based on similarity score
            best_experiment = results_df.loc[results_df['similarity'].idxmax()]
            
            # Save summary
            with open(os.path.join(example_dir, "summary.json"), "w") as f:
                json.dump({
                    "best_experiment_by_similarity": {
                        "experiment": best_experiment['experiment'],
                        "similarity": float(best_experiment['similarity'])
                    },
                    "best_experiment_by_supporting_docs": {
                        "experiment": results_df.loc[results_df['supporting_docs'].idxmax()]['experiment'],
                        "supporting_docs": int(results_df.loc[results_df['supporting_docs'].idxmax()]['supporting_docs'])
                    }
                }, f, indent=2)
        
        # Save entity caches to disk if we're not using a shared extractor
        if not shared_extractor:
            extractor.save_caches()
        
        return {
            "example_id": example_id,
            "results": experiment_results,
            "timing_data": timing_data
        }
        
    except Exception as e:
        print(f"Error processing example {example_id}: {e}")
        traceback.print_exc()
        
        # Save error information
        error_dir = os.path.join(output_dir, f"example_{example_id}")
        os.makedirs(error_dir, exist_ok=True)
        
        with open(os.path.join(error_dir, "error.txt"), "w") as f:
            f.write(f"Error processing example {example_id}: {str(e)}\n\n")
            f.write(traceback.format_exc())
        
        return {
            "example_id": example_id,
            "error": str(e)
        }

# Define this function at the module level so it can be pickled
def process_and_update_wrapper(args_tuple):
    """
    Wrapper function to process a question and update progress
    
    Args:
        args_tuple: Tuple of (example_id, df, args, output_dir)
        
    Returns:
        Result dictionary
    """
    example_id, df, args, output_dir = args_tuple
    try:
        result = process_single_question(example_id, df, args, output_dir)
        return {
            "example_id": example_id,
            "results": result
        }
    except Exception as e:
        print(f"Exception processing example {example_id}: {e}")
        traceback.print_exc()
        return {
            "example_id": example_id,
            "error": str(e)
        }

def batch_process_questions(example_ids, df, args, max_parallel=4):
    """
    Process multiple questions in parallel
    
    Args:
        example_ids: List of example IDs to process
        df: DataFrame containing the dataset
        args: Command line arguments
        max_parallel: Maximum number of parallel processes
        
    Returns:
        Tuple of (results, output_directory)
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"batch_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a shared extractor for all processes
    shared_extractor = EntityExtractor(
        verbose=args.verbose, 
        model_name=args.model,
        use_cache=not args.disable_cache,
        dataset_type=args.dataset_type  # Pass dataset type to EntityExtractor
    )
    
    # Process examples
    results = []
    all_experiment_results = []
    
    # Use tqdm for progress tracking
    with tqdm(total=len(example_ids), desc="Overall batch progress") as pbar:
        # Process in batches to limit parallelism
        for i in range(0, len(example_ids), max_parallel):
            batch = example_ids[i:i+max_parallel]
            
            # Process this batch in parallel
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_parallel) as executor:
                # Submit tasks
                futures = {
                    executor.submit(
                        process_single_question, 
                        example_id, 
                        df, 
                        args, 
                        output_dir,
                        shared_extractor  # Pass the shared extractor instance
                    ): example_id for example_id in batch
                }
                
                # Process results as they complete
                for future in concurrent.futures.as_completed(futures):
                    example_id = futures[future]
                    try:
                        result = future.result()
                        results.append(result)
                        
                        # Extract experiment results for aggregation
                        if 'results' in result and isinstance(result['results'], list):
                            for exp_result in result['results']:
                                if isinstance(exp_result, dict):
                                    # Add example_id to each experiment result
                                    exp_result['example_id'] = example_id
                                    all_experiment_results.append(exp_result)
                        
                        print(f"Completed processing example {example_id}")
                    except Exception as e:
                        print(f"Error processing example {example_id}: {e}")
                        traceback.print_exc()
                    
                    # Update progress bar
                    pbar.update(1)
    
    # Save batch results
    with open(os.path.join(output_dir, "batch_results.json"), "w") as f:
        json.dump({
            "timestamp": timestamp,
            "num_examples": len(example_ids),
            "results": results
        }, f, indent=2)
    
    # Create a summary of experiment results
    if all_experiment_results:
        try:
            # Create a DataFrame
            all_results_df = pd.DataFrame(all_experiment_results)
            
            # Save to CSV
            all_results_df.to_csv(os.path.join(output_dir, "all_experiments.csv"), index=False)
            
            # Calculate statistics for each experiment
            experiment_stats = {}
            for experiment in all_results_df['experiment'].unique():
                experiment_data = all_results_df[all_results_df['experiment'] == experiment]
                
                if not experiment_data.empty:
                    # Calculate statistics
                    stats = {
                        'example_count': len(experiment_data),
                        'avg_similarity': experiment_data['similarity'].mean(),
                        'avg_execution_time': experiment_data['execution_time'].mean(),
                        'avg_entity_count': experiment_data['entity_count'].mean(),
                        'avg_relationship_count': experiment_data['relationship_count'].mean(),
                        'avg_reachable_docs': experiment_data['reachable_docs'].mean(),
                        'avg_supporting_docs': experiment_data['supporting_docs'].mean(),
                        # Add exact match statistics
                        'exact_match_count': sum(1 for _, row in experiment_data.iterrows() if row['exact_match']),
                        'exact_match_percentage': (sum(1 for _, row in experiment_data.iterrows() if row['exact_match']) / len(experiment_data)) * 100,
                        # Add partial match statistics
                        'partial_match_count': sum(1 for _, row in experiment_data.iterrows() if row.get('partial_match', False)),
                        'partial_match_percentage': (sum(1 for _, row in experiment_data.iterrows() if row.get('partial_match', False)) / len(experiment_data)) * 100,
                        # Add F1, precision, and recall statistics
                        'avg_f1_score': experiment_data.get('f1_score', pd.Series([0])).mean(),
                        'avg_precision': experiment_data.get('precision', pd.Series([0])).mean(),
                        'avg_recall': experiment_data.get('recall', pd.Series([0])).mean()
                    }
                    
                    experiment_stats[experiment] = stats
            
            # Save experiment statistics to CSV
            pd.DataFrame(list(experiment_stats.items()), columns=['experiment', 'statistics']).to_csv(os.path.join(output_dir, "experiment_statistics.csv"), index=False)
            
            print("\nExperiment Statistics:")
            print(pd.DataFrame(experiment_stats).to_string(index=False))
        except Exception as e:
            print(f"Error generating statistics: {e}")
            print("This may be due to unexpected result format.")
    
    # Save entity caches to disk after all processing
    shared_extractor.save_caches()
    
    return results, output_dir

def show_api_usage():
    """Display current API usage statistics"""
    rate_limiter = get_rate_limiter()
    stats = rate_limiter.get_usage_stats()
    
    print("\nAPI Usage Statistics:")
    print("=====================")
    
    for model, usage in stats.items():
        print(f"\nModel: {model}")
        print(f"  Minute: {usage['minute']['count']}/{usage['minute']['limit']} requests " +
              f"(resets in {usage['minute']['reset_in']:.1f}s)")
        print(f"  Day: {usage['day']['count']}/{usage['day']['limit']} requests " +
              f"(resets in {usage['day']['reset_in']/3600:.1f}h)")

def find_reachable_documents(G, question_entities, max_hops=2):
    """
    Find documents that are reachable from question entities in a bipartite graph
    
    Args:
        G: NetworkX bipartite graph (entities ↔ documents)
        question_entities: List of entities from the question
        max_hops: Maximum number of hops to traverse
        
    Returns:
        List of document IDs that are reachable from question entities
    """
    # Get all document and entity nodes
    doc_nodes = set(n for n in G.nodes() if G.nodes[n].get('type') == 'document')
    entity_nodes = set(n for n in G.nodes() if G.nodes[n].get('type') == 'entity')
    
    # Start with question entities that exist in the graph
    current_entities = set(entity for entity in question_entities if entity in entity_nodes)
    reachable_entities = set(current_entities)
    reachable_docs = set()
    
    if not current_entities:
        print(f"Warning: None of the question entities {question_entities} found in graph")
        return []
    
    # Perform breadth-first traversal for max_hops
    for hop in range(max_hops):
        # Find documents connected to current entities
        new_docs = set()
        for entity in current_entities:
            for neighbor in G.neighbors(entity):
                if neighbor in doc_nodes:
                    new_docs.add(neighbor)
        
        # Add new documents to reachable set
        reachable_docs.update(new_docs)
        
        # Find entities connected to the new documents
        new_entities = set()
        for doc in new_docs:
            for neighbor in G.neighbors(doc):
                if neighbor in entity_nodes and neighbor not in reachable_entities:
                    new_entities.add(neighbor)
        
        # If no new entities found, we can stop early
        if not new_entities:
            break
            
        # Update reachable entities and prepare for next hop
        reachable_entities.update(new_entities)
        current_entities = new_entities
    
    print(f"Found {len(reachable_docs)} documents reachable from {len(question_entities)} question entities in {hop + 1} hops")
    print(f"Question entities: {question_entities}")
    print(f"Reachable entities: {sorted(list(reachable_entities))}")
    
    return list(reachable_docs)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Process MuSiQue or HotpotQA dataset examples')
    
    # Add dataset type argument
    parser.add_argument('--dataset-type', type=str, default="musique", choices=["musique", "hotpotqa"],
                       help='Type of dataset to use (musique or hotpotqa)')
    
    # Add dataset path argument with appropriate default based on dataset type
    parser.add_argument('--dataset-path', type=str, default=None,
                       help='Path to the dataset file (defaults to standard path for selected dataset type)')
    
    parser.add_argument('--test-api', action='store_true', help='Test the DeepSeek API connection')
    parser.add_argument('--example-id', type=str, help='ID of the example to process', default='6')
    parser.add_argument('--list-examples', action='store_true', help='List available example IDs')
    parser.add_argument('--max-hops', type=int, default=5, help='Maximum number of hops for document retrieval')
    parser.add_argument('--max-workers', type=int, default=1, help='Maximum number of parallel workers for API calls')
    parser.add_argument('--skip-bipartite', action='store_true', help='Skip bipartite graph visualization')
    parser.add_argument('--skip-visualization', action='store_true', help='Skip all visualizations')
    parser.add_argument('--experiment', type=str, choices=['standard', 'fuzzy_matching', 'llm_merging', 'sequential_context', 'all_docs_baseline', 'supporting_docs_baseline', 'all'],
                       default='standard', help='Entity extraction experiment to run')
    parser.add_argument('--batch', action='store_true', help='Run batch processing on multiple examples')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of examples to process in batch mode')
    parser.add_argument('--max-parallel', type=int, default=5, help='Maximum number of examples to process in parallel')
    parser.add_argument('--example-ids', type=str, help='Comma-separated list of example IDs to process in batch mode')
    parser.add_argument('--random-sample', action='store_true', help='Use random sampling for batch processing')
    parser.add_argument('--max-api-concurrency', type=int, default=1, 
                       help='Maximum number of concurrent API requests across all processes')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output with detailed timing information')
    parser.add_argument('--profile', action='store_true', help='Enable profiling for performance analysis')
    parser.add_argument('--model', type=str, default="deepseek-chat", 
                       choices=["deepseek-chat", "gpt-4o-mini", "gpt-3.5-turbo", "gpt-4.1-mini", 
                               "local:TheBloke/Llama-2-7B-Chat-GPTQ",
                               "local:TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
                               "local:TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
                               "local:Qwen/Qwen2.5-0.5B",
                               "local:Qwen/Qwen2.5-1.5B",
                               "o4-mini"],
                       help='Model to use for API calls')
    parser.add_argument('--show-usage', action='store_true', help='Show current API usage statistics')
    parser.add_argument('--disable-cache', action='store_true', help='Disable caching of API responses')
    
    args = parser.parse_args()
    
    # Set default dataset path based on dataset type if not provided
    if args.dataset_path is None:
        if args.dataset_type.lower() == 'musique':
            args.dataset_path = 'datasets/musique/musique_ans_v1.0_dev.jsonl'
        else:  # hotpotqa
            args.dataset_path = 'datasets/hotpotqa/hotpot_dev_distractor_v1.json'
    
    # Set the global API concurrency limit
    set_max_concurrent_requests(args.max_api_concurrency)
    
    # Apply profiling if requested
    if args.profile:
        global process_single_question
        process_single_question = profile_func(process_single_question)
    
    # Test API if requested
    if args.test_api:
        print("Testing API connection...")
        success = test_api(model_name=args.model)
        if success:
            print("API connection successful!")
        else:
            print("API connection failed. Please check your API key and try again.")
            return
    
    if args.show_usage:
        show_api_usage()
        return
    
    try:
        # Load dataset using the new load_dataset function
        print(f"Loading {args.dataset_type} dataset...")
        df = load_dataset(args.dataset_type, args.dataset_path)
        
        # List examples if requested
        if args.list_examples:
            print("\nAvailable example IDs (first 20):")
            for i, example_id in enumerate(df['id'].head(20)):
                print(f"- {i}: {example_id}")
            return
        
        # Batch processing
        if args.batch:
            # Determine which examples to process
            if args.example_ids:
                # Use provided list of IDs
                example_ids = [int(x.strip()) for x in args.example_ids.split(',')]
            elif args.random_sample:
                # Random sample
                example_ids = df.sample(min(args.batch_size, len(df))).index.tolist()
            else:
                # Use stratified sampling to ensure balanced hop distribution
                print("Using stratified sampling based on hop counts...")
                example_ids = stratified_sample(df, min(args.batch_size, len(df)))
                
                # Print the hop distribution in the sample
                distribution = get_hop_distribution(df, example_ids)
                print("Hop distribution in sample:")
                for hop, percentage in sorted(distribution.items()):
                    print(f"  {hop}-hop: {percentage:.2f}%")
            
            print(f"Batch processing {len(example_ids)} examples with max {args.max_parallel} in parallel")
            print(f"Example IDs: {example_ids}")
            
            # Process the batch
            results, output_dir = batch_process_questions(
                example_ids, 
                df, 
                args, 
                max_parallel=args.max_parallel
            )
            
            print(f"\nResults saved to: {output_dir}")
            
        # Single example processing
        elif args.example_id:
            # Create output directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join("results", f"single_{timestamp}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Process the example
            result = process_single_question(int(args.example_id), df, args, output_dir)
            
            print(f"\nResults saved to: {output_dir}")
            
        else:
            print("No example ID provided. Use --example-id to specify an example to process.")
            print("Use --list-examples to see available example IDs.")
            print("Use --batch to process multiple examples.")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease specify the correct path to the dataset using --dataset-path")
        print("Examples:")
        print("  python main.py --dataset-type musique --dataset-path /path/to/musique_full_v1.0_dev.jsonl")
        print("  python main.py --dataset-type hotpotqa --dataset-path /path/to/hotpot_dev_distractor_v1.json")
        print("  python main.py --dataset-type hotpotqa --dataset-path /path/to/hotpot_dev_fullwiki_v1.json")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()