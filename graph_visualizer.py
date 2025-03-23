import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict, List, Any

def visualize_entity_document_graph(G, question_entities, reachable_docs=None):
    """Visualize the bipartite entity-document graph"""
    plt.figure(figsize=(14, 10))
    
    # Create layout
    pos = nx.spring_layout(G, seed=42)
    
    # Prepare node colors
    node_colors = []
    node_sizes = []
    
    for node in G.nodes():
        if G.nodes[node]['type'] == 'entity':
            if node in question_entities:
                # Question entities in red
                node_colors.append('red')
                node_sizes.append(800)
            else:
                # Other entities in blue
                node_colors.append('lightblue')
                node_sizes.append(600)
        else:  # Document nodes
            is_reachable = reachable_docs and node in reachable_docs
            is_supporting = G.nodes[node].get('is_supporting', False)
            
            if is_reachable and is_supporting:
                # Both reachable and supporting - purple
                node_colors.append('purple')
                node_sizes.append(1200)
            elif is_reachable:
                # Reachable documents in green
                node_colors.append('lightgreen')
                node_sizes.append(1000)
            elif is_supporting:
                # Supporting documents in yellow
                node_colors.append('yellow')
                node_sizes.append(1000)
            else:
                # Other documents in gray
                node_colors.append('lightgray')
                node_sizes.append(800)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
    
    # Draw labels
    # For entities, use the entity name
    entity_labels = {node: node for node in G.nodes() if G.nodes[node]['type'] == 'entity'}
    nx.draw_networkx_labels(G, pos, labels=entity_labels, font_size=10, font_family='sans-serif')
    
    # For documents, use the title
    doc_labels = {node: G.nodes[node]['title'] for node in G.nodes() if G.nodes[node]['type'] == 'document'}
    nx.draw_networkx_labels(G, pos, labels=doc_labels, font_size=8, font_family='sans-serif')
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Question Entity', markerfacecolor='red', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Other Entity', markerfacecolor='lightblue', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Reachable & Supporting', markerfacecolor='purple', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Reachable Document', markerfacecolor='lightgreen', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Supporting Document', markerfacecolor='yellow', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Other Document', markerfacecolor='lightgray', markersize=10)
    ]
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.axis('off')
    plt.tight_layout()
    plt.show()

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

def visualize_entity_relationship_graph(entity_graph, question_entities=None):
    """
    Visualize the entity relationship graph with extracted relationships
    
    Args:
        entity_graph: A directed graph where nodes are entities and edges represent relationships
        question_entities: List of entities from the question (to highlight)
    """
    plt.figure(figsize=(14, 10))
    
    # Create layout
    pos = nx.spring_layout(entity_graph, seed=42)
    
    # Get edge weights for thickness
    edge_weights = [entity_graph[u][v]['weight'] for u, v in entity_graph.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    normalized_weights = [2 * w / max_weight for w in edge_weights]
    
    # Prepare node colors
    node_colors = []
    node_sizes = []
    
    for node in entity_graph.nodes():
        if question_entities and node in question_entities:
            # Question entities in red
            node_colors.append('red')
            node_sizes.append(800)
        else:
            # Other entities in blue
            node_colors.append('lightblue')
            node_sizes.append(600)
    
    # Draw nodes
    nx.draw_networkx_nodes(entity_graph, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8)
    
    # Draw edges with varying thickness based on weight and with arrows
    nx.draw_networkx_edges(entity_graph, pos, width=normalized_weights, alpha=0.5, 
                          arrowstyle='->', arrowsize=15)
    
    # Draw labels
    nx.draw_networkx_labels(entity_graph, pos, font_size=10, font_family='sans-serif')
    
    # Create edge labels showing the primary relationship
    edge_labels = {}
    for u, v, d in entity_graph.edges(data=True):
        if d['relations']:
            # Use the first relation as the label (to avoid cluttering)
            edge_labels[(u, v)] = d['relations'][0]
    
    # Draw edge labels
    nx.draw_networkx_edge_labels(entity_graph, pos, edge_labels=edge_labels, font_size=8)
    
    # Create legend
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', label='Question Entity', markerfacecolor='red', markersize=10),
        plt.Line2D([0], [0], marker='o', color='w', label='Other Entity', markerfacecolor='lightblue', markersize=10),
    ]
    
    plt.legend(handles=legend_elements, loc='upper right')
    
    plt.title("Entity Relationship Graph")
    plt.axis('off')
    plt.tight_layout()
    plt.show()