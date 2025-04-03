import streamlit as st
import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import glob
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set page configuration
st.set_page_config(page_title="MuSiQue Experiment Dashboard", layout="wide")

# Title and description
st.title("MuSiQue Experiment Dashboard")

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
    f1 = f1_score(y_true, y_pred, zero_division=0)
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
        st.write(f"{results['execution_time']:.2f} seconds")

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
            execution_time = results.get("execution_time", sum(results.get("timing", {}).values()))
            
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
    
    # Try to load batch_results.json first
    batch_results_path = os.path.join("results", selected_batch, "batch_results.json")
    if os.path.exists(batch_results_path):
        with open(batch_results_path, "r") as f:
            batch_results = json.load(f)
        
        st.subheader("Batch Summary")
        st.write(f"Total examples: {batch_results.get('total', 0)}")
        st.write(f"Successful: {batch_results.get('successful', 0)}")
        st.write(f"Failed: {batch_results.get('failed', 0)}")
        
        # Extract experiment results from all examples
        all_experiment_results = []
        for example_result in batch_results.get('results', []):
            example_id = example_result.get('example_id')
            results = example_result.get('results', {}).get('results', [])
            
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
                        
                        # Add metrics to result
                        exp_result['supporting_and_reachable'] = len([doc for doc in reachable if doc in true_supporting])
                        exp_result['precision'] = metrics["precision"]
                        exp_result['recall'] = metrics["recall"]
                        exp_result['f1'] = metrics["f1"]
                        exp_result['accuracy'] = metrics["accuracy"]
                    except Exception as e:
                        st.warning(f"Error calculating metrics for example {example_id}, experiment {exp_result['experiment']}: {e}")
                
                all_experiment_results.append(exp_result)
        
        if all_experiment_results:
            # Create DataFrame
            results_df = pd.DataFrame(all_experiment_results)
            
            # Display experiment statistics
            st.subheader("Experiment Statistics")
            
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
                
                # Add document retrieval metrics if available
                for metric in ['supporting_and_reachable', 'precision', 'recall', 'f1', 'accuracy']:
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
                    'mean_precision': 'avg_precision',
                    'mean_recall': 'avg_recall',
                    'mean_f1': 'avg_f1',
                    'mean_accuracy': 'avg_accuracy'
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
                
                # Document retrieval metrics
                if 'avg_precision' in agg_stats.columns:
                    st.subheader("Document Retrieval Metrics")
                    
                    # Create a grouped bar chart using Plotly
                    metrics_df = agg_stats[['experiment', 'avg_precision', 'avg_recall', 'avg_f1', 'avg_accuracy']]
                    
                    # Rename columns for display
                    metrics_df = metrics_df.rename(columns={
                        'avg_precision': 'Precision',
                        'avg_recall': 'Recall',
                        'avg_f1': 'F1 Score',
                        'avg_accuracy': 'Accuracy'
                    })
                    
                    # Create the figure
                    fig = go.Figure()
                    
                    # Add bars for each metric
                    for metric in ['Precision', 'Recall', 'F1 Score', 'Accuracy']:
                        fig.add_trace(go.Bar(
                            x=metrics_df['experiment'],
                            y=metrics_df[metric],
                            name=metric,
                            text=metrics_df[metric].round(2),
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
            
            # Display timing data
            st.subheader("Timing Data by Experiment")
            
            # Extract timing data
            timing_data = {}
            for example_result in batch_results.get('results', []):
                example_timing = example_result.get('results', {}).get('timing_data', {})
                for exp_type, times in example_timing.items():
                    if exp_type not in timing_data:
                        timing_data[exp_type] = {}
                    
                    for step, time_value in times.items():
                        if step not in timing_data[exp_type]:
                            timing_data[exp_type][step] = []
                        
                        timing_data[exp_type][step].append(time_value)
            
            # Create timing summary
            timing_summary = {}
            for exp_type, steps in timing_data.items():
                timing_summary[exp_type] = {}
                for step, times in steps.items():
                    timing_summary[exp_type][step] = {
                        'mean': sum(times) / len(times),
                        'total': sum(times),
                        'min': min(times),
                        'max': max(times)
                    }
            
            # Display timing summary
            for exp_type, steps in timing_summary.items():
                st.write(f"**{exp_type.replace('_', ' ').title()}**")
                
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
    
    # Fall back to experiment_statistics.csv if batch_results.json is not available
    else:
        batch_stats_path = os.path.join("results", selected_batch, "experiment_statistics.csv")
        if os.path.exists(batch_stats_path):
            stats_df = pd.read_csv(batch_stats_path)
            st.dataframe(stats_df)
        else:
            st.warning("Batch statistics not found.")