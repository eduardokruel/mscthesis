import argparse
import os
import sys
import json
from datetime import datetime
import time
import cProfile
import pstats
import io
from data_loader import load_musique_dataset, get_example_by_id
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

def process_single_question(example_id, df, args, output_dir):
    """
    Process a single question and save results
    
    Args:
        example_id: ID of the example to process
        df: DataFrame containing the dataset
        args: Command line arguments
        output_dir: Directory to save results
        
    Returns:
        Dictionary with experiment results
    """
    try:
        print(f"\nProcessing example with ID: {example_id}")
        
        # Create a directory for this example
        example_dir = os.path.join(output_dir, f"example_{example_id}")
        os.makedirs(example_dir, exist_ok=True)
        
        # Get the example
        example = get_example_by_id(df, example_id)
        
        # Save basic information
        with open(os.path.join(example_dir, "question_info.json"), "w") as f:
            json.dump({
                "id": example_id,
                "question": example["question"],
                "answer": example["answer"]
            }, f, indent=2)
        
        print(f"\nQuestion: {example['question']}")
        print(f"Answer: {example['answer']}")
        
        # Determine which experiments to run
        experiments = ['standard', 'fuzzy_matching', 'llm_merging', 'sequential_context'] if args.experiment == 'all' else [args.experiment]
        
        # Store results for comparison
        experiment_results = []
        timing_data = {}
        
        # Initialize the entity extractor once
        extractor = EntityExtractor(
            verbose=args.verbose, 
            model_name=args.model,
            use_cache=not args.disable_cache
        )
        
        # Extract entities from question and paragraphs once for all experiments
        print("\nExtracting entities from question and paragraphs (shared across experiments)...")
        start_time = time.time()
        
        # Extract question entities
        question_entities = extractor.extract_entities_from_question(example['question'])
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
                    futures.append(
                        executor.submit(
                            extractor.extract_entities_from_paragraph,
                            paragraph['paragraph_text']
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
        for experiment_type in experiments:
            print(f"\n{'='*50}")
            print(f"RUNNING EXPERIMENT: {experiment_type}")
            print(f"{'='*50}")
            
            # Create a directory for this experiment
            exp_dir = os.path.join(example_dir, experiment_type)
            os.makedirs(exp_dir, exist_ok=True)
            
            # Track timing for each major step
            timing = {}
            start_time = time.time()
            
            # Create bipartite graph based on experiment type
            print("\nCreating bipartite graph...")
            step_start = time.time()
            
            # Create a base graph with question entities
            G = nx.Graph()
            for entity in question_entities:
                G.add_node(entity, type='entity')
            
            # Apply the appropriate experiment
            if experiment_type == 'standard':
                G = extractor.apply_standard_experiment(G, example, question_entities, paragraph_entities)
            elif experiment_type == 'fuzzy_matching':
                G, mapped_question_entities = extractor.apply_fuzzy_matching_experiment(G, example, question_entities, paragraph_entities)
                question_entities = mapped_question_entities  # Use the mapped entities
            elif experiment_type == 'llm_merging':
                G, mapped_question_entities = extractor.apply_llm_merging_experiment(G, example, question_entities, paragraph_entities)
                question_entities = mapped_question_entities  # Use the mapped entities
            elif experiment_type == 'sequential_context':
                G, _ = extractor.apply_sequential_context_experiment(G, example, question_entities)
            
            timing['graph_creation'] = time.time() - step_start
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
            reachable_docs = extractor.find_reachable_documents(G, question_entities, max_hops=args.max_hops)
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
                step_start = time.time()
                fig = visualize_entity_document_graph(G, question_entities)
                # Save with both naming conventions for compatibility
                plt.savefig(os.path.join(exp_dir, "bipartite_graph.png"), dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(exp_dir, "entity_document_graph.png"), dpi=300, bbox_inches='tight')
                plt.close(fig)
                timing['visualization_bipartite'] = time.time() - step_start
                print(f"Bipartite graph visualization completed in {timing['visualization_bipartite']:.2f} seconds")
                
                print("\nVisualizing entity relationship graph...")
                step_start = time.time()
                fig = visualize_entity_relationship_graph(entity_graph)
                # Save with both naming conventions for compatibility
                plt.savefig(os.path.join(exp_dir, "relationship_graph.png"), dpi=300, bbox_inches='tight')
                plt.savefig(os.path.join(exp_dir, "entity_relationship_graph.png"), dpi=300, bbox_inches='tight')
                plt.close(fig)
                timing['visualization_relationships'] = time.time() - step_start
                print(f"Relationship graph visualization completed in {timing['visualization_relationships']:.2f} seconds")
            
            # Generate answer based on entity relationships and documents
            print("\nGenerating answer based on entity relationships and documents...")
            step_start = time.time()
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
            timing_data[experiment_type] = timing
            
            # Save answer and metrics
            with open(os.path.join(exp_dir, "results.json"), "w") as f:
                json.dump({
                    "generated_answer": generated_answer,
                    "expected_answer": example['answer'],
                    "similarity_score": sim_score,
                    "execution_time": execution_time,
                    "entity_count": entity_graph.number_of_nodes(),
                    "relationship_count": entity_graph.number_of_edges(),
                    "reachable_docs_count": len(reachable_docs),
                    "supporting_docs_count": len(supporting_docs)
                }, f, indent=2)
            
            # Store results for comparison
            experiment_results.append({
                'experiment': experiment_type,
                'reachable_docs': len(reachable_docs),
                'supporting_docs': len(supporting_docs),
                'entity_count': entity_graph.number_of_nodes(),
                'relationship_count': entity_graph.number_of_edges(),
                'answer': generated_answer,
                'similarity': sim_score,
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

def batch_process_questions(example_ids, df, args, max_parallel=3):
    """
    Process multiple questions in parallel
    
    Args:
        example_ids: List of example IDs to process
        df: DataFrame containing the dataset
        args: Command line arguments
        max_parallel: Maximum number of questions to process in parallel
        
    Returns:
        Tuple of (results, output_dir)
    """
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("results", f"batch_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save batch configuration
    with open(os.path.join(output_dir, "batch_config.json"), "w") as f:
        json.dump({
            "timestamp": timestamp,
            "example_ids": example_ids,
            "max_parallel": max_parallel,
            "experiment": args.experiment,
            "max_hops": args.max_hops,
            "max_workers": args.max_workers
        }, f, indent=2)
    
    # Create a progress bar for overall batch progress
    overall_progress = tqdm(total=len(example_ids), desc="Overall batch progress", position=0)
    
    # Process questions in parallel
    results = []
    
    # Prepare arguments for each task
    task_args = [(example_id, df, args, output_dir) for example_id in example_ids]
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_parallel) as executor:
        # Submit all tasks
        futures = [executor.submit(process_and_update_wrapper, arg) for arg in task_args]
        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                overall_progress.update(1)
                print(f"\nCompleted processing example {result['example_id']}")
            except Exception as e:
                print(f"Exception in future: {e}")
                traceback.print_exc()
    
    # Close the progress bar
    overall_progress.close()
    
    # Compile overall results
    success_count = sum(1 for r in results if "error" not in r)
    error_count = len(results) - success_count
    
    print(f"\n{'='*80}")
    print(f"BATCH PROCESSING COMPLETE")
    print(f"{'='*80}")
    print(f"Total examples processed: {len(results)}")
    print(f"Successful: {success_count}")
    print(f"Failed: {error_count}")
    
    # Save overall results
    with open(os.path.join(output_dir, "batch_results.json"), "w") as f:
        json.dump({
            "total": len(results),
            "successful": success_count,
            "failed": error_count,
            "results": results
        }, f, indent=2)
    
    # If we ran multiple experiments, create a comparison across all questions
    if args.experiment == 'all':
        # Collect all experiment results
        all_experiment_results = []
        
        for result in results:
            if "error" not in result and "results" in result:
                for exp_result in result["results"]:
                    # Create a copy of the experiment result to avoid modifying the original
                    exp_copy = dict(exp_result) if isinstance(exp_result, dict) else {"result": exp_result}
                    # Add the example_id to the copy
                    exp_copy["example_id"] = result["example_id"]
                    all_experiment_results.append(exp_copy)
        
        if all_experiment_results:
            # Create a DataFrame
            all_results_df = pd.DataFrame(all_experiment_results)
            
            # Save to CSV
            all_results_df.to_csv(os.path.join(output_dir, "all_experiments.csv"), index=False)
            
            # Calculate aggregate statistics by experiment type
            try:
                agg_stats = all_results_df.groupby('experiment').agg({
                    'similarity': ['mean', 'std', 'min', 'max'],
                    'supporting_docs': ['mean', 'sum'],
                    'execution_time': ['mean', 'sum'],
                    'example_id': 'count'
                }).reset_index()
                
                # Rename columns for clarity
                agg_stats.columns = [
                    'experiment', 
                    'avg_similarity', 'std_similarity', 'min_similarity', 'max_similarity',
                    'avg_supporting_docs', 'total_supporting_docs',
                    'avg_execution_time', 'total_execution_time',
                    'question_count'
                ]
                
                # Save aggregate statistics
                agg_stats.to_csv(os.path.join(output_dir, "experiment_statistics.csv"), index=False)
                
                print("\nExperiment Statistics:")
                print(agg_stats.to_string(index=False))
            except Exception as e:
                print(f"Error generating statistics: {e}")
                print("This may be due to unexpected result format.")
    
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

def main():
    parser = argparse.ArgumentParser(description='Extract entities from MuSiQue dataset and visualize relationships')
    parser.add_argument('--test-api', action='store_true', help='Test the DeepSeek API connection')
    parser.add_argument('--example-id', type=str, help='ID of the example to process', default='6')
    parser.add_argument('--dataset-path', type=str, help='Path to the MuSiQue dataset')
    parser.add_argument('--list-examples', action='store_true', help='List available example IDs')
    parser.add_argument('--max-hops', type=int, default=10, help='Maximum number of hops for document retrieval')
    parser.add_argument('--max-workers', type=int, default=1, help='Maximum number of parallel workers for API calls')
    parser.add_argument('--skip-bipartite', action='store_true', help='Skip bipartite graph visualization')
    parser.add_argument('--skip-visualization', action='store_true', help='Skip all visualizations')
    parser.add_argument('--experiment', type=str, choices=['standard', 'fuzzy_matching', 'llm_merging', 'sequential_context', 'all'],
                       default='standard', help='Entity extraction experiment to run')
    parser.add_argument('--batch', action='store_true', help='Run batch processing on multiple examples')
    parser.add_argument('--batch-size', type=int, default=5, help='Number of examples to process in batch mode')
    parser.add_argument('--max-parallel', type=int, default=1, help='Maximum number of examples to process in parallel')
    parser.add_argument('--example-ids', type=str, help='Comma-separated list of example IDs to process in batch mode')
    parser.add_argument('--random-sample', action='store_true', help='Use random sampling for batch processing')
    parser.add_argument('--max-api-concurrency', type=int, default=1, 
                       help='Maximum number of concurrent API requests across all processes')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output with detailed timing information')
    parser.add_argument('--profile', action='store_true', help='Enable profiling for performance analysis')
    parser.add_argument('--model', type=str, default="deepseek-chat", 
                       choices=["deepseek-chat", "gpt-4o-mini", "gpt-3.5-turbo", 
                               "local:TheBloke/Llama-2-7B-Chat-GPTQ",
                               "local:TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",
                               "local:TheBloke/Mixtral-8x7B-Instruct-v0.1-GPTQ",
                               "local:Qwen/Qwen2.5-0.5B",
                               "local:Qwen/Qwen2.5-1.5B"],
                       help='Model to use for API calls')
    parser.add_argument('--show-usage', action='store_true', help='Show current API usage statistics')
    parser.add_argument('--disable-cache', action='store_true', help='Disable caching of API responses')
    
    args = parser.parse_args()
    
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
        # Load dataset
        print("Loading dataset...")
        df = load_musique_dataset(args.dataset_path)
        
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
                # Sequential batch from the beginning
                example_ids = list(range(min(args.batch_size, len(df))))
            
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
        print("Example: python main.py --dataset-path /path/to/musique_full_v1.0_dev.jsonl")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()