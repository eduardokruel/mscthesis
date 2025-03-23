import argparse
import os
import sys
from data_loader import load_musique_dataset, get_example_by_id
from deepseek_api import test_api
from entity_extractor import EntityExtractor
from graph_visualizer import visualize_entity_document_graph, visualize_entity_relationship_graph
from difflib import SequenceMatcher

def main():
    parser = argparse.ArgumentParser(description='Extract entities from MuSiQue dataset and visualize relationships')
    parser.add_argument('--test-api', action='store_true', help='Test the DeepSeek API connection')
    parser.add_argument('--example-id', type=str, help='ID of the example to process', default='6')
    parser.add_argument('--dataset-path', type=str, help='Path to the MuSiQue dataset')
    parser.add_argument('--list-examples', action='store_true', help='List available example IDs')
    parser.add_argument('--max-hops', type=int, default=10, help='Maximum number of hops for document retrieval')
    parser.add_argument('--max-workers', type=int, default=20, help='Maximum number of parallel workers for API calls')
    parser.add_argument('--skip-bipartite', action='store_true', help='Skip bipartite graph visualization')
    
    args = parser.parse_args()
    
    # Test API if requested
    if args.test_api:
        print("Testing DeepSeek API connection...")
        success = test_api()
        if success:
            print("API connection successful!")
        else:
            print("API connection failed. Please check your API key and try again.")
            return
    
    try:
        # Load dataset
        print("Loading dataset...")
        df = load_musique_dataset(args.dataset_path)
        
        # List examples if requested
        if args.list_examples:
            print("\nAvailable example IDs (first 10):")
            for i, example_id in enumerate(df['id'].head(10)):
                print(f"- {i}: {example_id}")
            return
        
        # Process example if ID is provided
        if args.example_id:
            print(f"Processing example with ID: {args.example_id}")
            try:
                example = get_example_by_id(df, args.example_id)
                
                print(f"\nQuestion: {example['question']}")
                print(f"Answer: {example['answer']}")
                
                # Extract entities and create bipartite graph
                print("\nExtracting entities and creating bipartite graph...")
                extractor = EntityExtractor()
                G, question_entities = extractor.create_entity_document_graph(example, max_workers=args.max_workers)
                
                # Find reachable documents
                print(f"\nFinding documents reachable within {args.max_hops} hops...")
                reachable_docs = extractor.find_reachable_documents(G, question_entities, max_hops=args.max_hops)
                
                # Count supporting documents
                supporting_docs = [doc_id for doc_id in reachable_docs if G.nodes[doc_id].get('is_supporting', False)]
                
                print(f"\nFound {len(reachable_docs)} reachable documents, {len(supporting_docs)} of which are supporting.")
                
                # Print reachable and supporting documents first
                if supporting_docs:
                    print("\nReachable and supporting documents:")
                    for doc_id in supporting_docs:
                        doc_node = G.nodes[doc_id]
                        print(f"- {doc_id}: {doc_node['title']}")
                
                # Print reachable but not supporting documents
                non_supporting_reachable = [doc_id for doc_id in reachable_docs if doc_id not in supporting_docs]
                if non_supporting_reachable:
                    print("\nReachable but not supporting documents:")
                    for doc_id in non_supporting_reachable:
                        doc_node = G.nodes[doc_id]
                        print(f"- {doc_id}: {doc_node['title']}")
                
                # Create entity relationship graph based on reachable documents
                print("\nCreating entity relationship graph...")
                entity_graph = extractor.create_entity_relationship_graph(G, reachable_docs)
                
                print(f"Entity relationship graph has {entity_graph.number_of_nodes()} entities and {entity_graph.number_of_edges()} relationships")
                
                # Print the top relationships by weight
                edge_data = [(u, v, d) for u, v, d in entity_graph.edges(data=True)]
                edge_data.sort(key=lambda x: x[2]['weight'], reverse=True)
                
                if edge_data:
                    print("\nTop entity relationships:")
                    for u, v, d in edge_data[:10]:  # Show top 10
                        relations = d['relations']
                        relation_str = relations[0] if relations else "co-occurrence"
                        print(f"- {u} → {v}: {relation_str} (weight: {d['weight']})")
                        
                        # If there are multiple relations, show them
                        if len(relations) > 1:
                            print(f"  Additional relations:")
                            for rel in relations[1:3]:  # Show up to 3 additional relations
                                print(f"  • {rel}")
                            if len(relations) > 4:
                                print(f"  • ... and {len(relations) - 3} more")
                
                # Visualize the bipartite graph
                if not args.skip_bipartite:
                    print("\nVisualizing entity-document graph...")
                    visualize_entity_document_graph(G, question_entities, reachable_docs)
                
                # Visualize the entity relationship graph
                print("\nVisualizing entity relationship graph...")
                visualize_entity_relationship_graph(entity_graph, question_entities)
                
                # After visualizing the entity relationship graph
                print("\nGenerating answer based on entity relationships and documents...")
                generated_answer = extractor.generate_answer(
                    example['question'], 
                    entity_graph, 
                    question_entities, 
                    reachable_docs,
                    G
                )

                print("\n=== Generated Answer ===")
                print(generated_answer)
                print("\n=== Expected Answer ===")
                print(example['answer'])

                # Calculate simple similarity between generated and expected answers
                def similarity(a, b):
                    return SequenceMatcher(None, a, b).ratio()

                sim_score = similarity(generated_answer.lower(), example['answer'].lower())
                print(f"\nSimilarity score: {sim_score:.2f}")
                
            except IndexError as e:
                print(f"Error: {e}")
                print("Use --list-examples to see available example IDs.")
            except Exception as e:
                print(f"Error processing example: {e}")
        else:
            print("No example ID provided. Use --example-id to specify an example to process.")
            print("Use --list-examples to see available example IDs.")
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nPlease specify the correct path to the dataset using --dataset-path")
        print("Example: python main.py --dataset-path /path/to/musique_full_v1.0_dev.jsonl")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()