import networkx as nx
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Any

def visualize_entity_document_graph(G, question_entities=None, reachable_docs=None, save_path=None):
    """
    Create a visualization of the entity-document bipartite graph
    
    Args:
        G: NetworkX graph
        question_entities: List of entities from the question to highlight
        reachable_docs: List of document IDs that are reachable from question entities
        save_path: Path to save the visualization
        
    Returns:
        Matplotlib figure
    """
    # Create a new figure
    plt.figure(figsize=(12, 10))
    
    # Get entity and document nodes
    entity_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'entity']
    doc_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'document']
    
    # Create positions using spring layout
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # Draw entity nodes
    entity_colors = ['#ff9999' if n in question_entities else '#ff7f0e' for n in entity_nodes] if question_entities else ['#ff7f0e'] * len(entity_nodes)
    nx.draw_networkx_nodes(G, pos, nodelist=entity_nodes, node_color=entity_colors, node_size=500, alpha=0.8)
    
    # Draw document nodes with different colors for supporting documents
    supporting_docs = [n for n in doc_nodes if G.nodes[n].get('is_supporting', False)]
    non_supporting_docs = [n for n in doc_nodes if not G.nodes[n].get('is_supporting', False)]
    
    nx.draw_networkx_nodes(G, pos, nodelist=supporting_docs, node_color='#2ca02c', node_size=700, alpha=0.8, node_shape='s')
    nx.draw_networkx_nodes(G, pos, nodelist=non_supporting_docs, node_color='#1f77b4', node_size=700, alpha=0.8, node_shape='s')
    
    # Highlight reachable documents if provided
    if reachable_docs:
        reachable_nodes = [n for n in doc_nodes if n in reachable_docs]
        nx.draw_networkx_nodes(G, pos, nodelist=reachable_nodes, node_color='#1f77b4', 
                              node_size=700, alpha=0.8, node_shape='s', linewidths=3, edgecolors='red')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    # Draw labels with smaller font for better readability
    entity_labels = {n: n for n in entity_nodes}
    doc_labels = {n: n for n in doc_nodes}
    
    nx.draw_networkx_labels(G, pos, labels=entity_labels, font_size=8, font_weight='bold')
    nx.draw_networkx_labels(G, pos, labels=doc_labels, font_size=8)
    
    # Add legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=10, label='Entity'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#1f77b4', markersize=10, label='Document'),
        plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#2ca02c', markersize=10, label='Supporting Document')
    ]
    
    if question_entities:
        legend_elements.insert(0, plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff9999', markersize=10, label='Question Entity'))
    
    if reachable_docs:
        legend_elements.append(plt.Line2D([0], [0], marker='s', color='w', markerfacecolor='#1f77b4', markersize=10, 
                                         label='Reachable Document', markeredgecolor='red', markeredgewidth=2))
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved bipartite graph visualization to {save_path}")
    
    # Also save as entity_document_graph.png in the same directory
    if save_path:
        dir_path = os.path.dirname(save_path)
        standard_path = os.path.join(dir_path, "entity_document_graph.png")
        plt.savefig(standard_path, dpi=300, bbox_inches='tight')
        print(f"Saved bipartite graph visualization to {standard_path}")
    
    return plt.gcf()

def create_entity_graph(entities_data):
    """Create a graph from entity data (original function kept for compatibility)"""
    G = nx.DiGraph()
    
    # Add nodes (entities)
    for entity in entities_data.get('entities', []):
        G.add_node(entity['name'], type=entity['type'])
    
    # Add edges (relationships)
    for entity in entities_data.get('entities', []):
        for relationship in entity.get('relationships', []):
            G.add_edge(
                entity['name'], 
                relationship['related_entity'], 
                relation=relationship['relation_type']
            )
    
    return G

def visualize_graph(G):
    """Visualize the entity graph (original function kept for compatibility)"""
    plt.figure(figsize=(12, 8))
    
    # Define node colors based on entity type
    node_colors = []
    node_types = nx.get_node_attributes(G, 'type')
    
    color_map = {
        'person': 'lightblue',
        'location': 'lightgreen',
        'organization': 'lightcoral',
        'date': 'yellow',
        'event': 'orange'
    }
    
    for node in G.nodes():
        node_type = node_types.get(node, 'unknown')
        node_colors.append(color_map.get(node_type.lower(), 'gray'))
    
    # Create layout
    pos = nx.spring_layout(G, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=700, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=10, font_family='sans-serif')
    
    # Draw edge labels
    edge_labels = nx.get_edge_attributes(G, 'relation')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
    
    # Create legend
    legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                 label=entity_type.capitalize(),
                                 markerfacecolor=color, markersize=10)
                      for entity_type, color in color_map.items()]
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

def visualize_entity_relationship_graph(G, question_entities=None, save_path=None):
    """
    Create a visualization of the entity relationship graph
    
    Args:
        G: NetworkX graph
        question_entities: List of entities from the question to highlight
        save_path: Path to save the visualization
        
    Returns:
        Matplotlib figure
    """
    # Create a new figure
    plt.figure(figsize=(12, 10))
    
    # Create positions using spring layout
    pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
    
    # Draw nodes with different colors for question entities
    if question_entities:
        question_nodes = [n for n in G.nodes() if n in question_entities]
        other_nodes = [n for n in G.nodes() if n not in question_entities]
        
        nx.draw_networkx_nodes(G, pos, nodelist=question_nodes, node_color='#ff9999', node_size=500, alpha=0.8)
        nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_color='#ff7f0e', node_size=500, alpha=0.8)
    else:
        nx.draw_networkx_nodes(G, pos, node_color='#ff7f0e', node_size=500, alpha=0.8)
    
    # Draw edges with arrows
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True, arrowsize=15)
    
    # Draw labels
    nx.draw_networkx_labels(G, pos, font_size=8, font_weight='bold')
    
    # Draw edge labels (relationships)
    edge_labels = {(u, v): d.get('relation', '') for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
    
    # Add legend if question entities are highlighted
    if question_entities:
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff9999', markersize=10, label='Question Entity'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=10, label='Other Entity')
        ]
        plt.legend(handles=legend_elements, loc='upper right')
    
    plt.axis('off')
    plt.tight_layout()
    
    # Save the figure if a path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved relationship graph visualization to {save_path}")
    
    # Also save as entity_relationship_graph.png in the same directory
    if save_path:
        dir_path = os.path.dirname(save_path)
        standard_path = os.path.join(dir_path, "entity_relationship_graph.png")
        plt.savefig(standard_path, dpi=300, bbox_inches='tight')
        print(f"Saved relationship graph visualization to {standard_path}")
    
    return plt.gcf()