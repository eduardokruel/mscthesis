# Import required libraries
import json
import pandas as pd
from typing import List, Dict, Any, Set
import numpy as np
import os
import sys

# Function to load JSONL file
def load_jsonl(file_path):
    data = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data
    except FileNotFoundError:
        print(f"Error: File not found at '{file_path}'")
        raise

def get_absolute_path(relative_path):
    """Convert a relative path to an absolute path based on the current working directory"""
    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Get the project root directory (parent of the script directory)
    project_root = os.path.dirname(script_dir)
    
    # Try different base directories
    possible_bases = [
        os.getcwd(),           # Current working directory
        script_dir,            # Directory of this script
        project_root,          # Project root directory
        os.path.dirname(project_root)  # Parent of project root
    ]
    
    # Try to find the file using each base directory
    for base in possible_bases:
        abs_path = os.path.join(base, relative_path)
        if os.path.exists(abs_path):
            return abs_path
    
    # If we get here, we couldn't find the file
    return None

def load_musique_dataset(file_path=None):
    """Load and process the MuSiQue dataset"""
    # Default path if none provided
    if file_path is None:
        file_path = 'datasets/musique/musique_ans_v1.0_dev.jsonl'
    
    # Try to get absolute path
    abs_path = get_absolute_path(file_path)
    
    if abs_path is None:
        # If we couldn't find the file, try some alternative paths
        alternative_paths = [
            'mscthesis/datasets/musique/musique_ans_v1.0_dev.jsonl',
            '../datasets/musique/musique_ans_v1.0_dev.jsonl'
        ]
        
        for alt_path in alternative_paths:
            abs_path = get_absolute_path(alt_path)
            if abs_path is not None:
                break
    
    if abs_path is None:
        print(f"Error: Could not find dataset file at '{file_path}' or any alternative locations")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
        raise FileNotFoundError(f"Could not find dataset file: {file_path}")
    
    print(f"Loading dataset from: {abs_path}")
    
    # Load the development set
    dev_data = load_jsonl(abs_path)
    
    # Create a DataFrame for better visualization
    df = pd.DataFrame(dev_data)
    
    print(f"Dataset size: {len(df)}")
    print(f"\nColumns: {df.columns.tolist()}")
    
    # Calculate some statistics about paragraphs
    paragraph_counts = df['paragraphs'].apply(len)
    supporting_paragraph_counts = df['paragraphs'].apply(
        lambda paragraphs: sum(1 for p in paragraphs if p.get('is_supporting', False))
    )
    
    print("\nParagraph Statistics:")
    print(f"Average number of paragraphs per example: {paragraph_counts.mean()}")
    print(f"Average number of supporting paragraphs: {supporting_paragraph_counts.mean()}")
    print(f"Average number of supporting paragraphs: {df['answerable'].value_counts()}")
    
    return df

def get_example_by_id(df, example_id):
    """Get a specific example by ID"""
    print(df.iloc[int(example_id)])
    return df.iloc[int(example_id)]

def format_example_for_display(example):
    """Format an example for display"""
    formatted = {
        'id': example['id'],
        'question': example['question'],
        'answer': example['answer'],
        'paragraphs': [
            {
                'idx': p['idx'],
                'title': p['title'],
                'text': p['paragraph_text'],
                'is_supporting': p.get('is_supporting', False)
            }
            for p in example['paragraphs']
        ],
        'question_decomposition': example.get('question_decomposition', [])
    }
    return formatted 