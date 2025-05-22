import json
import os
import re
from collections import defaultdict

def load_jsonl(file_path):
    """Load a JSONL file and return a list of dictionaries."""
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        raise

def load_relationship_cache(cache_path):
    """Load the relationship cache from the specified path."""
    try:
        with open(cache_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Error: Relationship cache file not found at '{cache_path}'")
        raise

def normalize_text(text):
    """Normalize text for comparison by removing punctuation and converting to lowercase."""
    if not text:
        return ""
    
    # Handle case where text is a list
    if isinstance(text, list):
        # Convert list to string by joining elements
        text = " ".join(str(item) for item in text)
    elif not isinstance(text, str):
        # Convert any other non-string type to string
        text = str(text)
    
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def check_answer_in_relationships(answer, relationships):
    """Check if the answer or any of its aliases are present in the relationships."""
    if not relationships:
        return False
    
    normalized_answer = normalize_text(answer)
    
    # Check different relationship formats in the cache
    for rel_item in relationships:
        # Format 1: List of triples [entity1, entity2, relation]
        if isinstance(rel_item, list) and len(rel_item) == 3:
            entity1, entity2, _ = rel_item
            if (normalized_answer in normalize_text(entity1) or 
                normalized_answer in normalize_text(entity2)):
                return True
        
        # Format 2: Dictionary with source, target, relation
        elif isinstance(rel_item, dict) and 'source' in rel_item and 'target' in rel_item:
            source = rel_item.get('source', '')
            target = rel_item.get('target', '')
            if (normalized_answer in normalize_text(source) or 
                normalized_answer in normalize_text(target)):
                return True
    
    return False

def analyze_dataset(dataset_path, cache_path):
    """Analyze the dataset and check if answers are in the relationship cache."""
    # Load the dataset and relationship cache
    dataset = load_jsonl(dataset_path)
    relationship_cache = load_relationship_cache(cache_path)
    
    results = {
        'total_examples': len(dataset),
        'examples_with_relationships': 0,
        'examples_with_answer_in_relationships': 0,
        'detailed_results': []
    }
    
    for i, example in enumerate(dataset):
        example_id = example.get('id', f"example_{i}")
        question = example.get('question', '')
        answer = example.get('answer', '')
        
        # Track relationships for this example
        example_relationships = []
        answer_found = False
        
        # Check each paragraph for relationships
        for j, paragraph in enumerate(example.get('paragraphs', [])):
            doc_id = f"example_{i}_doc_{j}"
            rel_key = f"doc_relationships_{doc_id}"
            
            # Check if relationships exist for this document
            if rel_key in relationship_cache:
                relationships = relationship_cache[rel_key]
                example_relationships.append({
                    'doc_id': doc_id,
                    'relationships': relationships
                })
                
                # Check if answer is in relationships
                if check_answer_in_relationships(answer, relationships):
                    answer_found = True
        
        # Record results for this example
        example_result = {
            'example_id': example_id,
            'question': question,
            'answer': answer,
            'has_relationships': len(example_relationships) > 0,
            'answer_in_relationships': answer_found,
            'num_docs_with_relationships': len(example_relationships)
        }
        results['detailed_results'].append(example_result)
        
        if len(example_relationships) > 0:
            results['examples_with_relationships'] += 1
        
        if answer_found:
            results['examples_with_answer_in_relationships'] += 1
    
    # Calculate percentages
    if results['total_examples'] > 0:
        results['percent_with_relationships'] = (results['examples_with_relationships'] / results['total_examples']) * 100
        results['percent_with_answer_in_relationships'] = (results['examples_with_answer_in_relationships'] / results['total_examples']) * 100
        
        if results['examples_with_relationships'] > 0:
            results['percent_with_answer_among_those_with_relationships'] = (
                results['examples_with_answer_in_relationships'] / results['examples_with_relationships']) * 100
        else:
            results['percent_with_answer_among_those_with_relationships'] = 0
    
    return results

def main():
    # Define paths
    dataset_type = "musique"  # Default to musique, can be changed to hotpotqa
    
    if dataset_type.lower() == "musique":
        dataset_path = 'mscthesis/datasets/musique/musique_ans_v1.0_dev.jsonl'
    else:
        dataset_path = 'mscthesis/datasets/hotpotqa/hotpot_dev_distractor_v1.json'
    
    # Use dataset-specific cache
    cache_path = f'cache/{dataset_type.lower()}/relationship_cache.json'
    output_path = f'results/relationship_analysis_{dataset_type.lower()}.json'
    
    # Make sure the output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Analyzing dataset: {dataset_path}")
    print(f"Using relationship cache: {cache_path}")
    
    # Analyze the dataset
    results = analyze_dataset(dataset_path, cache_path)
    
    # Print summary results
    print("\nAnalysis Results:")
    print(f"Total examples: {results['total_examples']}")
    print(f"Examples with relationships: {results['examples_with_relationships']} ({results.get('percent_with_relationships', 0):.2f}%)")
    print(f"Examples with answer in relationships: {results['examples_with_answer_in_relationships']} ({results.get('percent_with_answer_in_relationships', 0):.2f}%)")
    
    if results['examples_with_relationships'] > 0:
        print(f"Percentage of examples with relationships that have answer in relationships: {results.get('percent_with_answer_among_those_with_relationships', 0):.2f}%")
    
    # Save detailed results to file
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nDetailed results saved to: {output_path}")

if __name__ == "__main__":
    main() 