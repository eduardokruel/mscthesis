import streamlit as st
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import glob
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score as sklearn_f1_score, accuracy_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import nltk
import re
from collections import defaultdict

# Set page configuration
st.set_page_config(page_title="MuSiQue Experiment Dashboard", layout="wide")

# Title and description
st.title("MuSiQue Experiment Dashboard")

# Add this near the top of the file, after imports
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

def identify_hop_count(example_id):
    """
    Identify the hop count from an example ID.
    
    Args:
        example_id: The ID of the example (e.g., "2hop__123456_789012")
        
    Returns:
        int: The number of hops (2, 3, or 4) or None if not identifiable
    """
    if isinstance(example_id, str):
        # Try to extract hop count from the ID format
        match = re.match(r'(\d+)hop', example_id)
        if match:
            return int(match.group(1))
    
    return None

def calculate_metrics_by_hop(all_experiment_results, batch_path=None):
    """
    Calculate F1, recall, and precision metrics grouped by hop count
    
    Args:
        all_experiment_results: List of experiment result dictionaries
        batch_path: Path to the batch directory to determine dataset type
        
    Returns:
        Dictionary with metrics grouped by hop count and experiment
    """
    # Import here to avoid circular imports
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))
    from data_loader import load_dataset
    
    # Try to determine dataset type from the batch results
    dataset_type = "musique"  # default
    
    if batch_path:
        # Try to read dataset type from a question_info.json file in the batch
        batch_dir = os.path.join("results", batch_path)
        if os.path.exists(batch_dir):
            # Look for any example directory to get dataset type
            example_dirs = [d for d in os.listdir(batch_dir) if d.startswith("example_")]
            if example_dirs:
                question_info_path = os.path.join(batch_dir, example_dirs[0], "question_info.json")
                if os.path.exists(question_info_path):
                    try:
                        with open(question_info_path, "r") as f:
                            question_info = json.load(f)
                            dataset_type = question_info.get("dataset_type", "musique")
                    except Exception:
                        pass  # Keep default
    
    # Try to load the dataset to get actual example IDs
    try:
        df = load_dataset(dataset_type)
        print(f"Successfully loaded {dataset_type} dataset with {len(df)} examples")
    except Exception as e:
        print(f"Warning: Could not load dataset: {e}")
        df = None
    
    # Group results by hop count and experiment
    hop_groups = defaultdict(lambda: defaultdict(list))
    
    for result in all_experiment_results:
        example_id = result.get('example_id')
        experiment = result.get('experiment')
        
        # Get hop count from example ID
        hop_count = None
        
        if df is not None and example_id is not None:
            try:
                # Convert example_id to int (it's the index) and get actual ID from dataset
                idx = int(example_id)
                if 0 <= idx < len(df):
                    actual_example_id = df.iloc[idx].get('id', '')
                    hop_count = identify_hop_count(str(actual_example_id))
                    if hop_count:
                        print(f"Example {idx}: {actual_example_id} -> {hop_count}-hop")
            except (ValueError, IndexError) as e:
                print(f"Error getting hop count for example {example_id}: {e}")
        
        if hop_count is None:
            hop_count = 'unknown'
        
        hop_groups[hop_count][experiment].append(result)
    
    # Print summary of hop distribution
    hop_counts = {}
    for hop_count, experiments in hop_groups.items():
        total_examples = sum(len(results) for results in experiments.values())
        hop_counts[hop_count] = total_examples
    
    print(f"Hop distribution: {hop_counts}")
    
    # Calculate metrics for each hop count and experiment
    metrics_by_hop = {}
    
    for hop_count, experiments in hop_groups.items():
        metrics_by_hop[hop_count] = {}
        
        for experiment, results in experiments.items():
            if not results:
                continue
                
            # Extract metrics from results
            f1_scores = [r.get('f1_score', 0) for r in results if 'f1_score' in r]
            precision_scores = [r.get('precision', 0) for r in results if 'precision' in r]
            recall_scores = [r.get('recall', 0) for r in results if 'recall' in r]
            exact_matches = [r.get('exact_match', False) for r in results if 'exact_match' in r]
            
            # Calculate average metrics
            metrics_by_hop[hop_count][experiment] = {
                'example_count': len(results),
                'avg_f1_score': np.mean(f1_scores) if f1_scores else 0,
                'avg_precision': np.mean(precision_scores) if precision_scores else 0,
                'avg_recall': np.mean(recall_scores) if recall_scores else 0,
                'exact_match_count': sum(exact_matches) if exact_matches else 0,
                'exact_match_percentage': (sum(exact_matches) / len(exact_matches) * 100) if exact_matches else 0,
                'std_f1_score': np.std(f1_scores) if len(f1_scores) > 1 else 0,
                'std_precision': np.std(precision_scores) if len(precision_scores) > 1 else 0,
                'std_recall': np.std(recall_scores) if len(recall_scores) > 1 else 0
            }
    
    return metrics_by_hop

# Function to load results directory
def load_results_directory():
    results_dir = "results"
    batch_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d)) and d.startswith("batch_")]
    batch_dirs.sort(reverse=True)  # Most recent first
    return batch_dirs

# Function to calculate document retrieval metrics
def calculate_doc_metrics(reachable_docs, supporting_docs, all_docs):
    """
    Calculate precision, recall, F1, and accuracy for document retrieval
    
    Args:
        reachable_docs: List of document IDs that were retrieved
        supporting_docs: List of document IDs that are actually supporting
        all_docs: List of all document IDs
        
    Returns:
        Dictionary with metrics
    """
    # Create binary arrays for metrics calculation
    y_true = np.zeros(len(all_docs))
    y_pred = np.zeros(len(all_docs))
    
    # Map document IDs to indices
    doc_to_idx = {doc_id: i for i, doc_id in enumerate(all_docs)}
    
    # Set true positives in ground truth
    for doc_id in supporting_docs:
        if doc_id in doc_to_idx:
            y_true[doc_to_idx[doc_id]] = 1
    
    # Set predicted positives
    for doc_id in reachable_docs:
        if doc_id in doc_to_idx:
            y_pred[doc_to_idx[doc_id]] = 1
    
    # Calculate metrics
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    # Rename the imported function to avoid conflict
    f1 = sklearn_f1_score(y_true, y_pred, zero_division=0)
    accuracy = accuracy_score(y_true, y_pred)
    
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy
    }

# Sidebar for navigation
st.sidebar.header("Navigation")

# Load available batch directories
batch_dirs = load_results_directory()
if not batch_dirs:
    st.error("No batch results found. Please run experiments first.")
    st.stop()

# Select batch
selected_batch = st.sidebar.selectbox("Select Batch", batch_dirs)

# Load examples in the selected batch
example_dir = os.path.join("results", selected_batch)
example_dirs = [d for d in os.listdir(example_dir) if os.path.isdir(os.path.join(example_dir, d)) and d.startswith("example_")]

if not example_dirs:
    st.error(f"No examples found in batch {selected_batch}. Please select another batch.")
    st.stop()

example_ids = [d.split("_")[1] for d in example_dirs]

# Select example - ensure we have a default value
if example_ids:
    default_example = example_ids[0]  # Use the first example as default
    selected_example = st.sidebar.selectbox("Select Example", example_ids, index=0)
else:
    st.error("No examples found in the selected batch.")
    st.stop()

# Load experiments for the selected example
experiment_dir = os.path.join("results", selected_batch, f"example_{selected_example}")

if not os.path.exists(experiment_dir):
    st.error(f"Example directory not found: {experiment_dir}")
    st.stop()

experiment_types = [d for d in os.listdir(experiment_dir) if os.path.isdir(os.path.join(experiment_dir, d))]

if not experiment_types:
    st.error(f"No experiments found for example {selected_example}.")
    st.stop()

# Move experiment selection to sidebar
st.sidebar.header("Experiment")
selected_experiment = st.sidebar.radio(
    "Select Experiment", 
    experiment_types,
    format_func=lambda x: x.replace('_', ' ').title()
)

# Main content
st.header(f"Example {selected_example}")

# Load question info
question_info_path = os.path.join("results", selected_batch, f"example_{selected_example}", "question_info.json")
if os.path.exists(question_info_path):
    with open(question_info_path, "r") as f:
        question_info = json.load(f)
    
    st.subheader("Question")
    st.write(question_info.get("question", "Question not found"))
    
    st.subheader("Reference Answer")
    st.write(question_info.get("answer", "Answer not found"))
    
    # Display dataset-specific information
    dataset_type = question_info.get("dataset_type", "musique")
    st.write(f"**Dataset:** {dataset_type.upper()}")
    
    if dataset_type.lower() == 'hotpotqa':
        # Display HotpotQA specific fields
        if 'type' in question_info:
            st.write(f"**Question Type:** {question_info['type']}")
        if 'level' in question_info:
            st.write(f"**Difficulty Level:** {question_info['level']}")

# Display selected experiment
exp_dir = os.path.join(experiment_dir, selected_experiment)

# Load results
results_path = os.path.join(exp_dir, "results.json")
if os.path.exists(results_path):
    with open(results_path, "r") as f:
        results = json.load(f)
    
    # Display generated answer - check multiple possible field names
    st.subheader("Generated Answer")
    answer = results.get("generated_answer", results.get("answer", "Answer not found"))
    st.write(answer)
    
    # Add exact match information
    if "exact_match" in results:
        exact_match = results["exact_match"]
        st.subheader(f"Exact Match: {'Yes' if exact_match else 'No'}")
        
        # Display extracted answer if available
        if "extracted_answer" in results:
            st.write(f"**Extracted Answer:** {results['extracted_answer']}")
        
        # Display normalized forms
        if "normalized_prediction" in results and "normalized_reference" in results:
            st.write(f"**Normalized Prediction:** {results['normalized_prediction']}")
            st.write(f"**Normalized Reference:** {results['normalized_reference']}")
        
        # Add display of best matching alias if available
        if "best_matching_alias" in results and results["best_matching_alias"] != results.get("reference_answer", ""):
            st.write(f"**Best Matching Alias:** {results['best_matching_alias']}")
            if "best_matching_alias_normalized" in results:
                st.write(f"**Normalized Alias:** {results['best_matching_alias_normalized']}")

    # Display similarity score - check multiple possible field names
    similarity = results.get("similarity_score", results.get("similarity", 0))
    st.subheader(f"Similarity Score: {similarity:.2f}")
    st.progress(similarity)
    
    # Display timing information
    timing = results.get("timing", {})
    if timing:
        st.subheader("Execution Time")
        timing_df = pd.DataFrame({
            "Step": list(timing.keys()),
            "Time (seconds)": list(timing.values())
        })
        st.bar_chart(timing_df.set_index("Step"))
    
    # Display total execution time if available
    if "execution_time" in results:
        st.subheader("Total Execution Time")
        timing_dict = results.get("timing", {})
        if timing_dict and all(isinstance(v, (int, float)) for v in timing_dict.values()):
            execution_time = results.get("execution_time", sum(timing_dict.values()))
        else:
            execution_time = results.get("execution_time", 0)
        st.write(f"{execution_time:.2f} seconds")

    # Add a new section to display all accepted aliases
    if "reference_answer" in results:
        reference_answer = results["reference_answer"]
        
        try:
            # Create an evaluator to get aliases
            from evaluation import AnswerEvaluator
            evaluator = AnswerEvaluator()
            aliases = evaluator.get_aliases(reference_answer)
            
            if aliases and len(aliases) > 1:  # Only show if there are aliases beyond the original answer
                with st.expander("View Accepted Answer Aliases"):
                    st.write("The following alternative forms of the answer are accepted:")
                    for alias in aliases:
                        if alias != reference_answer:  # Don't show the original answer again
                            st.write(f"- {alias}")
        except Exception as e:
            st.warning(f"Could not load answer aliases: {str(e)}")

    # After the exact match section
    if "partial_match" in results:
        partial_match = results["partial_match"]
        if partial_match and not exact_match:  # Only show if it's a partial match but not an exact match
            st.subheader(f"Partial Match: Yes")
            
            # Display F1 score if available
            if "f1_score" in results:
                f1_score = results["f1_score"]
                st.write(f"**F1 Score:** {f1_score:.2f}")
                
                # Show precision and recall
                if "precision" in results and "recall" in results:
                    st.write(f"**Precision:** {results['precision']:.2f}")
                    st.write(f"**Recall:** {results['recall']:.2f}")
            
            # Display substring relationship if available
            if "is_substring" in results and results["is_substring"]:
                st.write("**Note:** One answer is a substring of the other")

    # Add F1 score visualization
    if "f1_score" in results:
        f1_score = results["f1_score"]
        st.subheader(f"F1 Score: {f1_score:.2f}")
        
        # Create a progress bar for F1 score
        st.progress(f1_score)
        
        # Add color coding based on partial match criteria
        if f1_score > 0.8:
            st.success("High token overlap (F1 > 0.8)")
        elif f1_score > 0.6 and results.get("is_substring", False):
            st.success("Good token overlap (F1 > 0.6) with substring relationship")
        elif f1_score > 0.5:
            st.warning("Moderate token overlap")
        else:
            st.error("Low token overlap")

# Create tabs for visualizations
viz_tabs = st.tabs(["Entity-Document Graph", "Entity Relationship Graph", "Document Analysis"])

# Entity-Document Graph tab
with viz_tabs[0]:
    st.subheader("Entity-Document Bipartite Graph")
    # Try multiple possible filenames
    bipartite_img_paths = [
        os.path.join(exp_dir, "entity_document_graph.png"),
        os.path.join(exp_dir, "bipartite_graph.png")
    ]
    
    found_image = False
    for img_path in bipartite_img_paths:
        if os.path.exists(img_path):
            st.image(img_path)
            found_image = True
            break
    
    if not found_image:
        st.warning("Entity-document graph visualization not found.")

# Entity Relationship Graph tab
with viz_tabs[1]:
    st.subheader("Entity Relationship Graph")
    # Try multiple possible filenames
    entity_img_paths = [
        os.path.join(exp_dir, "entity_relationship_graph.png"),
        os.path.join(exp_dir, "relationship_graph.png")
    ]
    
    found_image = False
    for img_path in entity_img_paths:
        if os.path.exists(img_path):
            st.image(img_path)
            found_image = True
            break
    
    if not found_image:
        st.warning("Entity relationship graph visualization not found.")

# Document Analysis tab
with viz_tabs[2]:
    st.subheader("Document Analysis")
    reachable_path = os.path.join(exp_dir, "reachable_docs.json")
    if os.path.exists(reachable_path):
        with open(reachable_path, "r") as f:
            reachable_data = json.load(f)
        
        reachable = reachable_data.get("reachable_docs", [])
        supporting = reachable_data.get("supporting_docs", [])
        
        # Get document classification data
        doc_classification_path = os.path.join(exp_dir, "doc_classification.json")
        if os.path.exists(doc_classification_path):
            with open(doc_classification_path, "r") as f:
                doc_classification = json.load(f)
            
            true_positives = doc_classification.get("true_positives", [])
            false_negatives = doc_classification.get("false_negatives", [])
            false_positives = doc_classification.get("false_positives", [])
            true_negatives = doc_classification.get("true_negatives", [])
            
            # Calculate metrics directly
            precision = len(true_positives) / (len(true_positives) + len(false_positives)) if (len(true_positives) + len(false_positives)) > 0 else 0
            recall = len(true_positives) / (len(true_positives) + len(false_negatives)) if (len(true_positives) + len(false_negatives)) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (len(true_positives) + len(true_negatives)) / (len(true_positives) + len(false_positives) + len(false_negatives) + len(true_negatives)) if (len(true_positives) + len(false_positives) + len(false_negatives) + len(true_negatives)) > 0 else 0
            
            # Get all documents
            all_docs = true_positives + false_negatives + false_positives + true_negatives
            reachable = true_positives + false_positives
            true_supporting = true_positives + false_negatives
            
            # Get document info from bipartite graph if available
            bipartite_json_path = os.path.join(exp_dir, "bipartite_graph.json")
            if os.path.exists(bipartite_json_path):
                with open(bipartite_json_path, "r") as f:
                    graph_data = json.load(f)
                
                doc_info = {}
                for node in graph_data["nodes"]:
                    if node.get("type") == "document":
                        doc_id = node["id"]
                        doc_info[doc_id] = {
                            "title": node.get("title", ""),
                            "is_supporting": node.get("is_supporting", False)
                        }
            else:
                # Create basic doc info
                doc_info = {doc_id: {"title": f"Document {doc_id}", "is_supporting": doc_id in true_supporting} for doc_id in all_docs}
        else:
            # Try to extract document info from reachable_docs.json if bipartite_graph.json is missing
            st.warning("Bipartite graph data not found. Limited document analysis available.")
            all_docs = reachable + supporting
            true_supporting = supporting
            doc_info = {doc_id: {"title": f"Document {doc_id}", "is_supporting": doc_id in supporting} for doc_id in all_docs}
        
        # Calculate document categories
        supporting_and_reachable = [doc for doc in reachable if doc in true_supporting]
        supporting_not_reachable = [doc for doc in true_supporting if doc not in reachable]
        reachable_not_supporting = [doc for doc in reachable if doc not in true_supporting]
        
        # Display document counts
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Documents", len(all_docs))
        with col2:
            st.metric("True Supporting", len(true_supporting))
        with col3:
            st.metric("Reachable", len(reachable))
        with col4:
            st.metric("Supporting & Reachable", len(supporting_and_reachable))
        
        # Calculate and display metrics
        if all_docs:
            metrics = calculate_doc_metrics(reachable, true_supporting, all_docs)
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Precision", f"{metrics['precision']:.2f}")
            with col2:
                st.metric("Recall", f"{metrics['recall']:.2f}")
            with col3:
                st.metric("F1 Score", f"{metrics['f1']:.2f}")
            with col4:
                st.metric("Accuracy", f"{metrics['accuracy']:.2f}")
        
        # Create a table of all documents with their categories
        st.subheader("Document Details")
        
        if all_docs:
            docs_df = pd.DataFrame([
                {
                    "Document ID": doc_id,
                    "Title": doc_info.get(doc_id, {}).get("title", ""),
                    "Is Supporting (True Label)": doc_id in true_supporting,
                    "Is Reachable (Predicted)": doc_id in reachable,
                    "Category": "Supporting & Reachable" if doc_id in supporting_and_reachable else
                              ("Supporting (Not Reachable)" if doc_id in supporting_not_reachable else
                              ("Reachable (Not Supporting)" if doc_id in reachable_not_supporting else
                              "Not Supporting & Not Reachable"))
                }
                for doc_id in all_docs
            ])
            
            # Add color coding to the category column
            def color_category(val):
                if val == "Supporting & Reachable":
                    return 'background-color: #8eff8e'  # Light green
                elif val == "Supporting (Not Reachable)":
                    return 'background-color: #ffff8e'  # Light yellow
                elif val == "Reachable (Not Supporting)":
                    return 'background-color: #ff8e8e'  # Light red
                else:
                    return ''
            
            st.dataframe(docs_df.style.applymap(color_category, subset=['Category']))
    else:
        st.warning("Document analysis data not found.")

# Add comparison view
if st.sidebar.checkbox("Show Experiment Comparison", value=True):
    st.header("Experiment Comparison")
    
    # Load results for all experiments
    comparison_data = {}
    for exp_type in experiment_types:
        results_path = os.path.join(experiment_dir, exp_type, "results.json")
        reachable_path = os.path.join(experiment_dir, exp_type, "reachable_docs.json")
        bipartite_json_path = os.path.join(experiment_dir, exp_type, "bipartite_graph.json")
        
        if os.path.exists(results_path) and os.path.exists(reachable_path) and os.path.exists(bipartite_json_path):
            with open(results_path, "r") as f:
                results = json.load(f)
            
            with open(reachable_path, "r") as f:
                reachable_data = json.load(f)
            
            with open(bipartite_json_path, "r") as f:
                graph_data = json.load(f)
            
            # Get reachable docs
            reachable = reachable_data.get("reachable_docs", [])
            
            # Get true supporting docs
            true_supporting = []
            all_docs = []
            for node in graph_data["nodes"]:
                if node.get("type") == "document":
                    doc_id = node["id"]
                    all_docs.append(doc_id)
                    if node.get("is_supporting", False):
                        true_supporting.append(doc_id)
            
            # Calculate metrics
            metrics = calculate_doc_metrics(reachable, true_supporting, all_docs)
            
            # Check for different field names
            similarity = results.get("similarity_score", results.get("similarity", 0))
            timing_dict = results.get("timing", {})
            if timing_dict and all(isinstance(v, (int, float)) for v in timing_dict.values()):
                execution_time = results.get("execution_time", sum(timing_dict.values()))
            else:
                execution_time = results.get("execution_time", 0)
            
            comparison_data[exp_type] = {
                "similarity": similarity,
                "execution_time": execution_time,
                "reachable_docs": len(reachable),
                "supporting_docs": len(true_supporting),
                "supporting_and_reachable": len([doc for doc in reachable if doc in true_supporting]),
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "accuracy": metrics["accuracy"]
            }
    
    if comparison_data:
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_data).T
        comparison_df.index.name = "Experiment"
        comparison_df.reset_index(inplace=True)
        
        # Display comparison charts
        st.subheader("Answer Quality")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Similarity Score")
            chart_data = comparison_df[["Experiment", "similarity"]]
            chart_data = chart_data.rename(columns={"similarity": "Similarity"})
            st.bar_chart(chart_data.set_index("Experiment"))
        
        with col2:
            st.subheader("Execution Time")
            chart_data = comparison_df[["Experiment", "execution_time"]]
            chart_data = chart_data.rename(columns={"execution_time": "Execution Time (s)"})
            st.bar_chart(chart_data.set_index("Experiment"))
        
        st.subheader("Document Retrieval")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Document Counts")
            counts_df = comparison_df[["Experiment", "reachable_docs", "supporting_docs", "supporting_and_reachable"]]
            counts_df = counts_df.rename(columns={
                "reachable_docs": "Reachable", 
                "supporting_docs": "Supporting",
                "supporting_and_reachable": "Supporting & Reachable"
            })
            
            # Create the figure
            fig = go.Figure()
            
            # Add bars for each count
            for count in ['Reachable', 'Supporting', 'Supporting & Reachable']:
                fig.add_trace(go.Bar(
                    x=counts_df['Experiment'],
                    y=counts_df[count],
                    name=count,
                    text=counts_df[count],
                    textposition='auto'
                ))
            
            # Update layout - use STACKED mode
            fig.update_layout(
                title='Document Counts by Experiment',
                xaxis_title='Experiment',
                yaxis_title='Count',
                barmode='stack',  # Changed to stack
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            
            # Display the figure
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Retrieval Metrics")
            chart_data = comparison_df[["Experiment", "precision", "recall", "f1", "accuracy"]]
            chart_data = chart_data.rename(columns={
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1 Score",
                "accuracy": "Accuracy"
            })
            
            # Create grouped bar chart using plotly
            fig = px.bar(
                chart_data,
                x="Experiment",
                y=["Precision", "Recall", "F1 Score", "Accuracy"],
                barmode="group",
                title="Retrieval Metrics Comparison"
            )
            st.plotly_chart(fig, use_container_width=True, key="individual_retrieval_metrics_chart")
        
        # Display full comparison table
        st.subheader("Full Comparison")
        st.dataframe(comparison_df)

# Add batch statistics if available
if st.sidebar.checkbox("Show Batch Statistics", value=True):
    st.header("Batch Statistics")
    
    # Load batch_results.json if available
    batch_results_path = os.path.join("results", selected_batch, "batch_results.json")
    batch_results_available = False
    
    if os.path.exists(batch_results_path):
        with open(batch_results_path, "r") as f:
            batch_results = json.load(f)
        
        batch_results_available = True
        
        st.subheader("Batch Summary")
        st.write(f"Total examples: {batch_results.get('total', 0)}")
        st.write(f"Successful: {batch_results.get('successful', 0)}")
        st.write(f"Failed: {batch_results.get('failed', 0)}")
        
        # Extract experiment results from all examples
        all_experiment_results = []
        for example_result in batch_results.get('results', []):
            example_id = example_result.get('example_id')
            if isinstance(example_result, dict):
                example_timing = example_result.get('timing_data', {})
                results = example_result.get('results', [])
            elif isinstance(example_result, list):
                # If it's a list, we don't have timing data in this format
                example_timing = {}
                results = example_result
            else:
                example_timing = {}
                results = []
            
            for exp_result in results:
                exp_result['example_id'] = example_id
                
                # Add document retrieval metrics if available
                exp_dir = os.path.join("results", selected_batch, f"example_{example_id}", exp_result['experiment'])
                reachable_path = os.path.join(exp_dir, "reachable_docs.json")
                bipartite_json_path = os.path.join(exp_dir, "bipartite_graph.json")
                
                if os.path.exists(reachable_path) and os.path.exists(bipartite_json_path):
                    try:
                        with open(reachable_path, "r") as f:
                            reachable_data = json.load(f)
                        
                        with open(bipartite_json_path, "r") as f:
                            graph_data = json.load(f)
                        
                        # Get reachable docs
                        reachable = reachable_data.get("reachable_docs", [])
                        
                        # Get true supporting docs
                        true_supporting = []
                        all_docs = []
                        for node in graph_data["nodes"]:
                            if node.get("type") == "document":
                                doc_id = node["id"]
                                all_docs.append(doc_id)
                                if node.get("is_supporting", False):
                                    true_supporting.append(doc_id)
                        
                        # Calculate metrics
                        metrics = calculate_doc_metrics(reachable, true_supporting, all_docs)
                        
                        # Add DOCUMENT RETRIEVAL metrics to result (with different names to avoid overriding answer metrics)
                        exp_result['supporting_and_reachable'] = len([doc for doc in reachable if doc in true_supporting])
                        exp_result['doc_precision'] = metrics["precision"]  # Document retrieval precision
                        exp_result['doc_recall'] = metrics["recall"]        # Document retrieval recall
                        exp_result['doc_f1'] = metrics["f1"]                # Document retrieval F1
                        exp_result['doc_accuracy'] = metrics["accuracy"]    # Document retrieval accuracy
                        
                        # Keep the original answer quality metrics (f1_score, precision, recall) as they were loaded from results.json
                    except Exception as e:
                        st.warning(f"Error calculating metrics for example {example_id}, experiment {exp_result['experiment']}: {e}")
                
                all_experiment_results.append(exp_result)
        
        if all_experiment_results:
            # Create DataFrame
            results_df = pd.DataFrame(all_experiment_results)
            
            # Display experiment statistics
            st.subheader("Experiment Statistics from Batch Results")
            
            # Group by experiment type
            if 'experiment' in results_df.columns:
                # Determine which metrics are available
                metric_columns = {
                    'similarity': ['mean', 'std', 'min', 'max'],
                    'supporting_docs': ['mean', 'sum'],
                    'reachable_docs': ['mean', 'sum'],
                    'execution_time': ['mean', 'sum'],
                    'example_id': ['count']
                }
                
                # Add answer quality metrics if available
                for metric in ['f1_score', 'precision', 'recall', 'exact_match', 'partial_match']:
                    if metric in results_df.columns:
                        metric_columns[metric] = ['mean', 'std', 'min', 'max']
                
                # Add document retrieval metrics if available
                for metric in ['supporting_and_reachable', 'doc_precision', 'doc_recall', 'doc_f1', 'doc_accuracy']:
                    if metric in results_df.columns:
                        metric_columns[metric] = ['mean', 'std', 'min', 'max']
                
                # Aggregate statistics
                agg_stats = results_df.groupby('experiment').agg(metric_columns).reset_index()
                
                # Flatten multi-level columns
                flat_columns = ['experiment']
                for col, aggs in metric_columns.items():
                    for agg in aggs:
                        flat_columns.append(f"{agg}_{col}")
                
                agg_stats.columns = flat_columns
                
                # Rename columns for clarity
                column_mapping = {
                    'count_example_id': 'question_count',
                    'mean_similarity': 'avg_similarity',
                    'std_similarity': 'std_similarity',
                    'min_similarity': 'min_similarity',
                    'max_similarity': 'max_similarity',
                    'mean_supporting_docs': 'avg_supporting_docs',
                    'sum_supporting_docs': 'total_supporting_docs',
                    'mean_reachable_docs': 'avg_reachable_docs',
                    'sum_reachable_docs': 'total_reachable_docs',
                    'mean_supporting_and_reachable': 'avg_supporting_and_reachable',
                    'sum_supporting_and_reachable': 'total_supporting_and_reachable',
                    'mean_execution_time': 'avg_execution_time',
                    'sum_execution_time': 'total_execution_time',
                    # Answer quality metrics
                    'mean_f1_score': 'avg_answer_f1',
                    'mean_precision': 'avg_answer_precision',
                    'mean_recall': 'avg_answer_recall',
                    'mean_exact_match': 'avg_exact_match',
                    'mean_partial_match': 'avg_partial_match',
                    # Document retrieval metrics  
                    'mean_doc_precision': 'avg_doc_precision',
                    'mean_doc_recall': 'avg_doc_recall',
                    'mean_doc_f1': 'avg_doc_f1',
                    'mean_doc_accuracy': 'avg_doc_accuracy'
                }
                
                agg_stats = agg_stats.rename(columns={old: new for old, new in column_mapping.items() if old in agg_stats.columns})
                
                st.dataframe(agg_stats)
                
                # Create charts for key metrics
                st.subheader("Key Metrics by Experiment")
                
                # Answer quality metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("Average Similarity")
                    if 'avg_similarity' in agg_stats.columns:
                        chart_data = agg_stats[['experiment', 'avg_similarity']]
                        st.bar_chart(chart_data.set_index('experiment'))
                
                with col2:
                    st.subheader("Average Execution Time")
                    if 'avg_execution_time' in agg_stats.columns:
                        chart_data = agg_stats[['experiment', 'avg_execution_time']]
                        st.bar_chart(chart_data.set_index('experiment'))
                
                # Answer quality metrics chart
                if 'avg_answer_f1' in agg_stats.columns:
                    st.subheader("Answer Quality Metrics")
                    
                    # Create a grouped bar chart for answer quality metrics
                    available_cols = ['experiment']
                    display_names = []
                    actual_cols = []
                    
                    if 'avg_answer_f1' in agg_stats.columns:
                        available_cols.append('avg_answer_f1')
                        display_names.append('F1 Score')
                        actual_cols.append('avg_answer_f1')
                    if 'avg_answer_precision' in agg_stats.columns:
                        available_cols.append('avg_answer_precision')
                        display_names.append('Precision')
                        actual_cols.append('avg_answer_precision')
                    if 'avg_answer_recall' in agg_stats.columns:
                        available_cols.append('avg_answer_recall')
                        display_names.append('Recall')
                        actual_cols.append('avg_answer_recall')
                    if 'avg_exact_match' in agg_stats.columns:
                        available_cols.append('avg_exact_match')
                        display_names.append('Exact Match Rate')
                        actual_cols.append('avg_exact_match')
                    
                    if len(actual_cols) > 0:
                        answer_metrics_df = agg_stats[available_cols]
                        
                        # Create the figure
                        fig = go.Figure()
                        
                        # Add bars for each metric
                        for i, col in enumerate(actual_cols):
                            fig.add_trace(go.Bar(
                                x=answer_metrics_df['experiment'],
                                y=answer_metrics_df[col],
                                name=display_names[i],
                                text=answer_metrics_df[col].round(3),
                                textposition='auto'
                            ))
                        
                        # Update layout
                        fig.update_layout(
                            title='Answer Quality Metrics by Experiment',
                            xaxis_title='Experiment',
                            yaxis_title='Score',
                            barmode='group',
                            yaxis=dict(range=[0, 1]),
                            legend=dict(
                                orientation="h",
                                yanchor="bottom",
                                y=1.02,
                                xanchor="right",
                                x=1
                            )
                        )
                        
                        # Display the figure
                        st.plotly_chart(fig, use_container_width=True, key="answer_quality_metrics_chart")
                
                # Document retrieval metrics
                if 'avg_doc_precision' in agg_stats.columns:
                    st.subheader("Document Retrieval Metrics")
                    
                    # Create a grouped bar chart using Plotly
                    metrics_df = agg_stats[['experiment', 'avg_doc_precision', 'avg_doc_recall', 'avg_doc_f1', 'avg_doc_accuracy']]
                    
                    # Rename columns for display
                    metrics_df = metrics_df.rename(columns={
                        'avg_doc_precision': 'Precision',
                        'avg_doc_recall': 'Recall',
                        'avg_doc_f1': 'F1 Score',
                        'avg_doc_accuracy': 'Accuracy'
                    })
                    
                    # Create the figure
                    fig = go.Figure()
                    
                    # Add bars for each metric
                    for metric in ['Precision', 'Recall', 'F1 Score', 'Accuracy']:
                        if metric in metrics_df.columns:
                            fig.add_trace(go.Bar(
                                x=metrics_df['experiment'],
                                y=metrics_df[metric],
                                name=metric,
                                text=metrics_df[metric].round(3),
                                textposition='auto'
                            ))
                    
                    # Update layout - explicitly set to GROUP mode
                    fig.update_layout(
                        title='Document Retrieval Metrics by Experiment',
                        xaxis_title='Experiment',
                        yaxis_title='Score',
                        barmode='group',  # Explicitly set to group
                        yaxis=dict(range=[0, 1]),
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    # Display the figure
                    st.plotly_chart(fig, use_container_width=True, key="retrieval_metrics_chart")
            
            # Add hop-based analysis if available
            if all_experiment_results:
                st.markdown("---")
                st.subheader("Answer Quality Metrics by Question Type (Hop Count)")
                
                # Calculate metrics by hop
                metrics_by_hop = calculate_metrics_by_hop(all_experiment_results, selected_batch)
                
                if metrics_by_hop:
                    # Create a comprehensive DataFrame for all metrics
                    hop_data = []
                    for hop_count, experiments in metrics_by_hop.items():
                        for experiment, metrics in experiments.items():
                            hop_data.append({
                                'hop_count': hop_count,
                                'experiment': experiment,
                                'example_count': metrics['example_count'],
                                'avg_f1_score': metrics['avg_f1_score'],
                                'avg_precision': metrics['avg_precision'],
                                'avg_recall': metrics['avg_recall'],
                                'exact_match_percentage': metrics['exact_match_percentage'],
                                'std_f1_score': metrics['std_f1_score'],
                                'std_precision': metrics['std_precision'],
                                'std_recall': metrics['std_recall']
                            })
                    
                    hop_df = pd.DataFrame(hop_data)
                    
                    if not hop_df.empty:
                        # Display summary table
                        st.subheader("Summary Table")
                        # Format the DataFrame for better display
                        display_df = hop_df.copy()
                        display_df['avg_f1_score'] = display_df['avg_f1_score'].round(3)
                        display_df['avg_precision'] = display_df['avg_precision'].round(3)
                        display_df['avg_recall'] = display_df['avg_recall'].round(3)
                        display_df['exact_match_percentage'] = display_df['exact_match_percentage'].round(1)
                        
                        # Rename columns for better display
                        display_df = display_df.rename(columns={
                            'hop_count': 'Hop Count',
                            'experiment': 'Experiment',
                            'example_count': 'Examples',
                            'avg_f1_score': 'Avg F1',
                            'avg_precision': 'Avg Precision',
                            'avg_recall': 'Avg Recall',
                            'exact_match_percentage': 'Exact Match %'
                        })
                        
                        st.dataframe(display_df[['Hop Count', 'Experiment', 'Examples', 'Avg F1', 'Avg Precision', 'Avg Recall', 'Exact Match %']])
                        
                        # Create visualizations
                        st.subheader("Answer Quality Metrics by Hop Count")
                        
                        # Create tabs for different visualizations
                        viz_tabs = st.tabs(["F1 Score", "Precision & Recall", "Exact Match Rate", "All Metrics"])
                        
                        with viz_tabs[0]:
                            # F1 Score by hop count and experiment
                            fig = px.bar(
                                hop_df, 
                                x='hop_count', 
                                y='avg_f1_score',
                                color='experiment',
                                title='Average F1 Score by Hop Count and Experiment',
                                labels={'avg_f1_score': 'Average F1 Score', 'hop_count': 'Hop Count'},
                                barmode='group',
                                text='avg_f1_score'
                            )
                            fig.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                            fig.update_layout(yaxis_range=[0, 1])
                            st.plotly_chart(fig, use_container_width=True)
                        
                        with viz_tabs[1]:
                            # Precision and Recall side by side
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                fig_precision = px.bar(
                                    hop_df, 
                                    x='hop_count', 
                                    y='avg_precision',
                                    color='experiment',
                                    title='Average Precision by Hop Count',
                                    labels={'avg_precision': 'Average Precision', 'hop_count': 'Hop Count'},
                                    barmode='group',
                                    text='avg_precision'
                                )
                                fig_precision.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                                fig_precision.update_layout(yaxis_range=[0, 1])
                                st.plotly_chart(fig_precision, use_container_width=True)
                            
                            with col2:
                                fig_recall = px.bar(
                                    hop_df, 
                                    x='hop_count', 
                                    y='avg_recall',
                                    color='experiment',
                                    title='Average Recall by Hop Count',
                                    labels={'avg_recall': 'Average Recall', 'hop_count': 'Hop Count'},
                                    barmode='group',
                                    text='avg_recall'
                                )
                                fig_recall.update_traces(texttemplate='%{text:.3f}', textposition='outside')
                                fig_recall.update_layout(yaxis_range=[0, 1])
                                st.plotly_chart(fig_recall, use_container_width=True)
                        
                        with viz_tabs[2]:
                            # Exact Match Rate
                            fig_exact = px.bar(
                                hop_df, 
                                x='hop_count', 
                                y='exact_match_percentage',
                                color='experiment',
                                title='Exact Match Percentage by Hop Count',
                                labels={'exact_match_percentage': 'Exact Match %', 'hop_count': 'Hop Count'},
                                barmode='group',
                                text='exact_match_percentage'
                            )
                            fig_exact.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                            fig_exact.update_layout(yaxis_range=[0, 100])
                            st.plotly_chart(fig_exact, use_container_width=True)
                        
                        with viz_tabs[3]:
                            # All metrics in one chart
                            # Create a melted DataFrame for easier plotting
                            metrics_melted = hop_df.melt(
                                id_vars=['hop_count', 'experiment'],
                                value_vars=['avg_f1_score', 'avg_precision', 'avg_recall'],
                                var_name='metric',
                                value_name='score'
                            )
                            
                            # Rename metrics for better display
                            metric_names = {
                                'avg_f1_score': 'F1 Score',
                                'avg_precision': 'Precision',
                                'avg_recall': 'Recall'
                            }
                            metrics_melted['metric'] = metrics_melted['metric'].map(metric_names)
                            
                            fig_all = px.bar(
                                metrics_melted,
                                x='hop_count',
                                y='score',
                                color='experiment',
                                facet_col='metric',
                                title='All Answer Quality Metrics by Hop Count',
                                labels={'score': 'Score', 'hop_count': 'Hop Count'},
                                barmode='group'
                            )
                            fig_all.update_layout(yaxis_range=[0, 1])
                            fig_all.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))
                            st.plotly_chart(fig_all, use_container_width=True)
                        
                        # Statistical insights
                        st.subheader("Statistical Insights")
                        
                        # Show how metrics change with hop count
                        hop_summary = hop_df.groupby('hop_count').agg({
                            'avg_f1_score': ['mean', 'std'],
                            'avg_precision': ['mean', 'std'],
                            'avg_recall': ['mean', 'std'],
                            'exact_match_percentage': ['mean', 'std'],
                            'example_count': 'sum'
                        }).round(3)
                        
                        # Flatten column names
                        hop_summary.columns = [f"{col[1]}_{col[0]}" if col[1] else col[0] for col in hop_summary.columns]
                        hop_summary = hop_summary.reset_index()
                        
                        st.write("**Average metrics across all experiments by hop count:**")
                        st.dataframe(hop_summary)
                        
                        # Best performing experiment by hop count
                        st.write("**Best performing experiment by hop count (based on F1 score):**")
                        best_by_hop = hop_df.loc[hop_df.groupby('hop_count')['avg_f1_score'].idxmax()]
                        best_display = best_by_hop[['hop_count', 'experiment', 'avg_f1_score', 'avg_precision', 'avg_recall', 'exact_match_percentage']].round(3)
                        st.dataframe(best_display)
                    else:
                        st.warning("No hop-based metrics data available.")
                else:
                    st.warning("No hop count information found in example IDs.")
            
            # Display timing data
            st.subheader("Timing Data by Experiment")
            
            # Display timing summary
            for exp_type, steps in example_timing.items():
                st.write(f"**{exp_type.replace('_', ' ').title()}**")
                
                # Check if the timing data is in the new format (direct float values)
                if isinstance(next(iter(steps.values()), None), (int, float)):
                    # New format - direct values
                    timing_df = pd.DataFrame([
                        {
                            'Step': step,
                            'Time (s)': time_value
                        }
                        for step, time_value in steps.items()
                    ])
                    
                    st.dataframe(timing_df)
                    
                    # Create bar chart of times
                    chart_data = pd.DataFrame({
                        'Step': list(steps.keys()),
                        'Time (s)': list(steps.values())
                    })
                    st.bar_chart(chart_data.set_index('Step'))
                else:
                    # Old format - dictionaries with mean, total, etc.
                    timing_df = pd.DataFrame([
                        {
                            'Step': step,
                            'Mean Time (s)': data['mean'],
                            'Total Time (s)': data['total'],
                            'Min Time (s)': data['min'],
                            'Max Time (s)': data['max']
                        }
                        for step, data in steps.items()
                    ])
                    
                    st.dataframe(timing_df)
                    
                    # Create bar chart of mean times
                    chart_data = pd.DataFrame({
                        'Step': list(steps.keys()),
                        'Mean Time (s)': [data['mean'] for data in steps.values()]
                    })
                    st.bar_chart(chart_data.set_index('Step'))
        
        # Display raw results
        if st.checkbox("Show Raw Batch Results", value=False):
            st.subheader("Raw Batch Results")
            st.json(batch_results)
    
    # Also check for experiment_statistics.csv
    batch_stats_path = os.path.join("results", selected_batch, "experiment_statistics.csv")
    if os.path.exists(batch_stats_path):
        # Read the statistics CSV file
        stats_df = pd.read_csv(batch_stats_path)
        
        # Add a separator if we already displayed batch_results.json data
        if batch_results_available:
            st.markdown("---")
            st.subheader("Additional Experiment Statistics")
        
        # Check if the statistics are in the new format (with 'statistics' column as JSON)
        if 'statistics' in stats_df.columns:
            # Parse the statistics JSON column
            try:
                # Convert the statistics column from string to dictionary
                stats_df['statistics'] = stats_df['statistics'].apply(lambda x: eval(x) if isinstance(x, str) else x)
                
                # Create a new DataFrame with the statistics expanded
                expanded_stats = pd.DataFrame()
                expanded_stats['experiment'] = stats_df['experiment']
                
                # Extract each statistic into its own column
                for stat in ['example_count', 'avg_similarity', 'avg_execution_time', 
                            'avg_entity_count', 'avg_relationship_count', 
                            'avg_reachable_docs', 'avg_supporting_docs',
                            'exact_match_count', 'exact_match_percentage',
                            'partial_match_count', 'partial_match_percentage',
                            'avg_f1_score', 'avg_precision', 'avg_recall']:
                    expanded_stats[stat] = stats_df['statistics'].apply(
                        lambda x: x.get(stat, 0) if isinstance(x, dict) else 0
                    )
                
                # Display the expanded statistics
                st.subheader("Experiment Statistics from CSV")
                st.dataframe(expanded_stats)
                
                # Create visualizations for key metrics
                st.subheader("Key Metrics by Experiment")
                
                # Create tabs for different metric categories
                metric_tabs = st.tabs(["Answer Quality", "Document Retrieval", "Entity Statistics"])
                
                # Answer Quality tab
                with metric_tabs[0]:
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Similarity score chart
                        st.subheader("Average Similarity")
                        fig = px.bar(
                            expanded_stats, 
                            x='experiment', 
                            y='avg_similarity',
                            title='Average Similarity by Experiment',
                            labels={'avg_similarity': 'Similarity', 'experiment': 'Experiment'},
                            text_auto='.2f'
                        )
                        fig.update_layout(yaxis_range=[0, 1])
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Exact match percentage chart
                        st.subheader("Exact Match Percentage")
                        fig = px.bar(
                            expanded_stats, 
                            x='experiment', 
                            y='exact_match_percentage',
                            title='Exact Match Percentage by Experiment',
                            labels={'exact_match_percentage': 'Exact Match %', 'experiment': 'Experiment'},
                            text_auto='.1f'
                        )
                        fig.update_layout(yaxis_range=[0, 100])
                        st.plotly_chart(fig, use_container_width=True)
                
                # Document Retrieval tab
                with metric_tabs[1]:
                    # Document retrieval metrics
                    st.subheader("Document Retrieval")
                    
                    # Create a figure with two y-axes
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Add bars for reachable docs
                    fig.add_trace(
                        go.Bar(
                            x=expanded_stats['experiment'],
                            y=expanded_stats['avg_reachable_docs'],
                            name='Avg. Reachable Docs',
                            marker_color='blue',
                            text=expanded_stats['avg_reachable_docs'].round(1)
                        ),
                        secondary_y=False
                    )
                    
                    # Add bars for supporting docs
                    fig.add_trace(
                        go.Bar(
                            x=expanded_stats['experiment'],
                            y=expanded_stats['avg_supporting_docs'],
                            name='Avg. Supporting Docs',
                            marker_color='green',
                            text=expanded_stats['avg_supporting_docs'].round(1)
                        ),
                        secondary_y=False
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title='Document Retrieval by Experiment',
                        barmode='group',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    # Set y-axes titles
                    fig.update_yaxes(title_text="Number of Documents", secondary_y=False)
                    
                    # Display the figure
                    st.plotly_chart(fig, use_container_width=True)
                
                # Entity Statistics tab
                with metric_tabs[2]:
                    # Entity and relationship counts
                    st.subheader("Entity and Relationship Statistics")
                    
                    # Create a figure with two y-axes
                    fig = make_subplots(specs=[[{"secondary_y": True}]])
                    
                    # Add bars for entity count
                    fig.add_trace(
                        go.Bar(
                            x=expanded_stats['experiment'],
                            y=expanded_stats['avg_entity_count'],
                            name='Avg. Entity Count',
                            marker_color='purple',
                            text=expanded_stats['avg_entity_count'].round(1)
                        ),
                        secondary_y=False
                    )
                    
                    # Add bars for relationship count
                    fig.add_trace(
                        go.Bar(
                            x=expanded_stats['experiment'],
                            y=expanded_stats['avg_relationship_count'],
                            name='Avg. Relationship Count',
                            marker_color='orange',
                            text=expanded_stats['avg_relationship_count'].round(1)
                        ),
                        secondary_y=False
                    )
                    
                    # Update layout
                    fig.update_layout(
                        title='Entity and Relationship Statistics by Experiment',
                        barmode='group',
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="right",
                            x=1
                        )
                    )
                    
                    # Set y-axes titles
                    fig.update_yaxes(title_text="Count", secondary_y=False)
                    
                    # Display the figure
                    st.plotly_chart(fig, use_container_width=True)
                
                # Execution time chart
                st.subheader("Average Execution Time")
                fig = px.bar(
                    expanded_stats, 
                    x='experiment', 
                    y='avg_execution_time',
                    title='Average Execution Time by Experiment',
                    labels={'avg_execution_time': 'Time (s)', 'experiment': 'Experiment'},
                    text_auto='.2f'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Add a new row for F1, precision, and recall
                st.subheader("Answer Evaluation Metrics")
                fig = px.bar(
                    expanded_stats, 
                    x='experiment', 
                    y=['avg_f1_score', 'avg_precision', 'avg_recall'],
                    title='Answer Evaluation Metrics by Experiment',
                    labels={
                        'avg_f1_score': 'F1 Score', 
                        'avg_precision': 'Precision', 
                        'avg_recall': 'Recall', 
                        'experiment': 'Experiment'
                    },
                    barmode='group'
                )
                fig.update_layout(yaxis_range=[0, 1])
                st.plotly_chart(fig, use_container_width=True)
            
            except Exception as e:
                st.error(f"Error parsing statistics: {e}")
                st.dataframe(stats_df)
        else:
            # Display the original statistics DataFrame
            st.dataframe(stats_df)
    elif not batch_results_available:
        st.warning("No batch statistics found.")