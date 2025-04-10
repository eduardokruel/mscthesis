# Import required libraries
import time
import json
import pandas as pd
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Optional
import os

# Import local modules
from data_loader import load_musique_dataset
from local_llm import LocalLLM

def create_entity_extraction_prompt(paragraph: str) -> List[Dict[str, str]]:
    """
    Create a prompt for entity extraction
    
    Args:
        paragraph: Text to extract entities from
        
    Returns:
        List of message dictionaries for the LLM
    """
    return [
        {
            "role": "system", 
            "content": "You are a helpful assistant that extracts entities from text."
        },
        {
            "role": "user", 
            "content": f"""
        Extract all entities from the following paragraph:
        
        Paragraph: {paragraph}
        
        An entity is a real-world object such as a person, location, organization, product, etc.
        Return only a JSON array of entity names, with no additional text.
        Example: ["Entity1", "Entity2", "Entity3"]
        """
        }
    ]

def run_benchmark(
    model_name: str,
    dataset_path: Optional[str] = None,
    num_samples: int = 100,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Run a benchmark for entity extraction
    
    Args:
        model_name: Name of the model to benchmark
        dataset_path: Path to the dataset
        num_samples: Number of samples to process
        verbose: Whether to print verbose output
        
    Returns:
        Dictionary with benchmark results
    """
    print(f"Running benchmark with model: {model_name}")
    
    # Load dataset
    df = load_musique_dataset(dataset_path)
    
    # Initialize model
    start_time = time.time()
    llm = LocalLLM(model_name_or_path=model_name, verbose=verbose)
    model_load_time = time.time() - start_time
    print(f"Model loaded in {model_load_time:.2f} seconds")
    
    # Select samples
    if num_samples < len(df):
        samples = df.sample(num_samples)
    else:
        samples = df
        print(f"Using all {len(df)} samples")
    
    # Prepare results
    results = {
        "model_name": model_name,
        "model_load_time": model_load_time,
        "num_samples": len(samples),
        "sample_times": [],
        "sample_results": [],
        "errors": 0
    }
    
    # Process samples
    for idx, row in tqdm(samples.iterrows(), total=len(samples), desc="Processing samples"):
        # Get paragraphs
        paragraphs = row['paragraphs']
        
        # Process first paragraph only for benchmark
        if paragraphs and len(paragraphs) > 0:
            paragraph = paragraphs[0]['text'] if 'text' in paragraphs[0] else paragraphs[0]['p']
            
            # Create prompt
            prompt = create_entity_extraction_prompt(paragraph)
            
            # Generate response
            try:
                start_time = time.time()
                response = llm.generate(prompt)
                end_time = time.time()
                
                # Parse response
                try:
                    entities = json.loads(response)
                    is_valid_json = True
                except:
                    entities = response
                    is_valid_json = False
                
                # Record result
                sample_result = {
                    "id": row['id'],
                    "paragraph": paragraph,
                    "response": response,
                    "entities": entities,
                    "is_valid_json": is_valid_json,
                    "time": end_time - start_time
                }
                
                results["sample_times"].append(end_time - start_time)
                results["sample_results"].append(sample_result)
                
                if verbose:
                    print(f"Sample {idx} processed in {end_time - start_time:.2f} seconds")
                    print(f"Entities: {entities}")
            except Exception as e:
                results["errors"] += 1
                if verbose:
                    print(f"Error processing sample {idx}: {e}")
    
    # Calculate statistics
    if results["sample_times"]:
        results["avg_time"] = sum(results["sample_times"]) / len(results["sample_times"])
        results["min_time"] = min(results["sample_times"])
        results["max_time"] = max(results["sample_times"])
        results["total_time"] = sum(results["sample_times"])
        
        # Calculate success rate for valid JSON
        valid_json_count = sum(1 for r in results["sample_results"] if r["is_valid_json"])
        results["valid_json_rate"] = valid_json_count / len(results["sample_results"]) if results["sample_results"] else 0
    
    return results

def save_benchmark_results(results: Dict[str, Any], output_dir: str = "benchmark_results"):
    """
    Save benchmark results to files
    
    Args:
        results: Benchmark results
        output_dir: Directory to save results
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    
    # Save summary
    summary = {
        "model_name": results["model_name"],
        "model_load_time": results["model_load_time"],
        "num_samples": results["num_samples"],
        "avg_time": results.get("avg_time", 0),
        "min_time": results.get("min_time", 0),
        "max_time": results.get("max_time", 0),
        "total_time": results.get("total_time", 0),
        "errors": results["errors"],
        "valid_json_rate": results.get("valid_json_rate", 0)
    }
    
    with open(f"{output_dir}/summary_{results['model_name'].replace('/', '_')}_{timestamp}.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    # Save detailed results
    with open(f"{output_dir}/details_{results['model_name'].replace('/', '_')}_{timestamp}.json", "w") as f:
        # Remove the full sample results to keep the file size manageable
        results_copy = results.copy()
        results_copy["sample_results"] = f"Saved separately ({len(results['sample_results'])} samples)"
        json.dump(results_copy, f, indent=2)
    
    # Save sample results separately
    with open(f"{output_dir}/samples_{results['model_name'].replace('/', '_')}_{timestamp}.json", "w") as f:
        json.dump(results["sample_results"], f, indent=2)
    
    # Generate plots
    if results.get("sample_times"):
        plt.figure(figsize=(10, 6))
        plt.hist(results["sample_times"], bins=20)
        plt.title(f"Processing Time Distribution - {results['model_name']}")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Frequency")
        plt.savefig(f"{output_dir}/time_hist_{results['model_name'].replace('/', '_')}_{timestamp}.png")
        
        plt.figure(figsize=(10, 6))
        plt.plot(results["sample_times"])
        plt.title(f"Processing Time per Sample - {results['model_name']}")
        plt.xlabel("Sample Index")
        plt.ylabel("Time (seconds)")
        plt.savefig(f"{output_dir}/time_plot_{results['model_name'].replace('/', '_')}_{timestamp}.png")
    
    print(f"Results saved to {output_dir}")
    return summary

def main():
    parser = argparse.ArgumentParser(description='Benchmark entity extraction performance')
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-1.5B", 
                       help='Model to benchmark')
    parser.add_argument('--dataset-path', type=str, help='Path to the MuSiQue dataset')
    parser.add_argument('--num-samples', type=int, default=100, 
                       help='Number of samples to process')
    parser.add_argument('--verbose', action='store_true', 
                       help='Enable verbose output')
    parser.add_argument('--output-dir', type=str, default="benchmark_results",
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Run benchmark
    results = run_benchmark(
        model_name=args.model,
        dataset_path=args.dataset_path,
        num_samples=args.num_samples,
        verbose=args.verbose
    )
    
    # Save results
    summary = save_benchmark_results(results, args.output_dir)
    
    # Print summary
    print("\nBenchmark Summary:")
    print(f"Model: {summary['model_name']}")
    print(f"Samples: {summary['num_samples']}")
    print(f"Model Load Time: {summary['model_load_time']:.2f} seconds")
    print(f"Average Processing Time: {summary['avg_time']:.2f} seconds")
    print(f"Min Processing Time: {summary['min_time']:.2f} seconds")
    print(f"Max Processing Time: {summary['max_time']:.2f} seconds")
    print(f"Total Processing Time: {summary['total_time']:.2f} seconds")
    print(f"Errors: {summary['errors']}")
    print(f"Valid JSON Rate: {summary['valid_json_rate'] * 100:.2f}%")

if __name__ == "__main__":
    main() 