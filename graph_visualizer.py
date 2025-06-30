import networkx as nx
import matplotlib.pyplot as plt
import os
from typing import Dict, List, Any
import matplotlib

def visualize_entity_document_graph(G, question_entities=None, reachable_docs=None, save_path=None, non_interactive=False):
    """
    Create a visualization of the entity-document bipartite graph
    
    Args:
        G: NetworkX bipartite graph
        question_entities: List of entities from the question to highlight
        reachable_docs: Set of document IDs that are reachable
        save_path: Path to save the visualization
        non_interactive: If True, use non-interactive backend (for parallel processing)
        
    Returns:
        Matplotlib figure or None if non_interactive or if error occurs
    """
    try:
    # Use non-interactive backend if requested (for parallel processing)
    if non_interactive:
            matplotlib.use('Agg')  # Set backend before importing pyplot
            
        import matplotlib.pyplot as plt
        
        # Create figure
            fig = plt.figure(figsize=(12, 10))
            
            # Get entity and document nodes
            entity_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'entity']
            doc_nodes = [n for n in G.nodes() if G.nodes[n].get('type') == 'document']
            
            # Create positions using spring layout
        try:
            pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
        except Exception as e:
            print(f"Warning: Spring layout failed, using random layout: {e}")
            pos = nx.random_layout(G, seed=42)
            
            # Draw entity nodes
        if entity_nodes:
            entity_colors = ['#ff9999' if n in (question_entities or []) else '#ff7f0e' for n in entity_nodes]
            nx.draw_networkx_nodes(G, pos, nodelist=entity_nodes, node_color=entity_colors, node_size=500, alpha=0.8)
            
            # Draw document nodes with different colors for supporting documents
        if doc_nodes:
            supporting_docs = [n for n in doc_nodes if G.nodes[n].get('is_supporting', False)]
            non_supporting_docs = [n for n in doc_nodes if not G.nodes[n].get('is_supporting', False)]
            
            if supporting_docs:
            nx.draw_networkx_nodes(G, pos, nodelist=supporting_docs, node_color='#2ca02c', node_size=700, alpha=0.8, node_shape='s')
            if non_supporting_docs:
            nx.draw_networkx_nodes(G, pos, nodelist=non_supporting_docs, node_color='#1f77b4', node_size=700, alpha=0.8, node_shape='s')
            
            # Highlight reachable documents if provided
            if reachable_docs:
                reachable_nodes = [n for n in doc_nodes if n in reachable_docs]
                if reachable_nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=reachable_nodes, node_color='#1f77b4', 
                                      node_size=700, alpha=0.8, node_shape='s', linewidths=3, edgecolors='red')
            
            # Draw edges
        if G.edges():
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5)
            
            # Draw labels with smaller font for better readability
        try:
            if entity_nodes:
                entity_labels = {n: str(n)[:20] + '...' if len(str(n)) > 20 else str(n) for n in entity_nodes}
            nx.draw_networkx_labels(G, pos, labels=entity_labels, font_size=8, font_weight='bold')
            
            if doc_nodes:
                doc_labels = {n: str(n)[:15] + '...' if len(str(n)) > 15 else str(n) for n in doc_nodes}
            nx.draw_networkx_labels(G, pos, labels=doc_labels, font_size=8)
        except Exception as e:
            print(f"Warning: Could not draw labels: {e}")
            
            # Add legend
        try:
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
        except Exception as e:
            print(f"Warning: Could not create legend: {e}")
            
            plt.axis('off')
            plt.tight_layout()
            
            # Save the figure
            if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                print(f"Saved bipartite graph visualization to {save_path}")
            except Exception as e:
                print(f"Warning: Could not save visualization to {save_path}: {e}")
                # Try saving with simpler parameters
                try:
                    plt.savefig(save_path, dpi=150)
                    print(f"Saved bipartite graph visualization to {save_path} (simplified)")
                except Exception as e2:
                    print(f"Error: Failed to save visualization: {e2}")
            
            # Close the figure to free memory
            plt.close(fig)
            
        # Return None for non-interactive mode
        if non_interactive:
            return None
    else:
        return fig
            
    except Exception as e:
        print(f"Error creating bipartite graph visualization: {e}")
        print("Continuing without visualization...")
        return None

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

def visualize_entity_relationship_graph(G, question_entities=None, save_path=None, non_interactive=False):
    """
    Create a visualization of the entity relationship graph
    
    Args:
        G: NetworkX graph
        question_entities: List of entities from the question to highlight
        save_path: Path to save the visualization
        non_interactive: If True, use non-interactive backend (for parallel processing)
        
    Returns:
        Matplotlib figure or None if non_interactive or if error occurs
    """
    try:
    # Use non-interactive backend if requested (for parallel processing)
    if non_interactive:
            import matplotlib
            matplotlib.use('Agg')  # Set backend before importing pyplot
            
        import matplotlib.pyplot as plt
        
        # Create figure
            fig = plt.figure(figsize=(12, 10))
            
        # Check if graph has nodes
        if not G.nodes():
            plt.text(0.5, 0.5, 'No entity relationships found', 
                    horizontalalignment='center', verticalalignment='center', 
                    transform=plt.gca().transAxes, fontsize=16)
            plt.axis('off')
            
            if save_path:
                try:
                    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                    print(f"Saved empty relationship graph to {save_path}")
                except Exception as e:
                    print(f"Warning: Could not save empty graph: {e}")
            
            plt.close(fig)
            return None if non_interactive else fig
        
            # Create positions using spring layout
        try:
            pos = nx.spring_layout(G, k=0.3, iterations=50, seed=42)
        except Exception as e:
            print(f"Warning: Spring layout failed, using random layout: {e}")
            try:
                pos = nx.random_layout(G, seed=42)
            except Exception as e2:
                print(f"Error: Could not create layout: {e2}")
                plt.close(fig)
                return None
            
            # Draw nodes with different colors for question entities
        try:
            if question_entities:
                question_nodes = [n for n in G.nodes() if n in question_entities]
                other_nodes = [n for n in G.nodes() if n not in question_entities]
                
                if question_nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=question_nodes, node_color='#ff9999', node_size=500, alpha=0.8)
                if other_nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, node_color='#ff7f0e', node_size=500, alpha=0.8)
            else:
                nx.draw_networkx_nodes(G, pos, node_color='#ff7f0e', node_size=500, alpha=0.8)
        except Exception as e:
            print(f"Warning: Could not draw nodes: {e}")
            
            # Draw edges with arrows
        try:
            if G.edges():
            nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, arrows=True, arrowsize=15)
        except Exception as e:
            print(f"Warning: Could not draw edges: {e}")
            
            # Draw labels
        try:
            node_labels = {n: str(n)[:15] + '...' if len(str(n)) > 15 else str(n) for n in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, font_weight='bold')
        except Exception as e:
            print(f"Warning: Could not draw node labels: {e}")
            
            # Draw edge labels (relationships)
        try:
            edge_labels = {}
            for u, v, d in G.edges(data=True):
                if 'relations' in d and d['relations']:
                    # Take the first relation if multiple exist
                    relation = d['relations'][0] if isinstance(d['relations'], list) else d['relations']
                    edge_labels[(u, v)] = str(relation)[:10] + '...' if len(str(relation)) > 10 else str(relation)
                elif 'relation' in d:
                    relation = str(d['relation'])
                    edge_labels[(u, v)] = relation[:10] + '...' if len(relation) > 10 else relation
            
            if edge_labels:
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=6)
        except Exception as e:
            print(f"Warning: Could not draw edge labels: {e}")
            
            # Add legend if question entities are highlighted
        try:
            if question_entities:
                legend_elements = [
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff9999', markersize=10, label='Question Entity'),
                    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#ff7f0e', markersize=10, label='Other Entity')
                ]
                plt.legend(handles=legend_elements, loc='upper right')
        except Exception as e:
            print(f"Warning: Could not create legend: {e}")
            
            plt.axis('off')
            plt.tight_layout()
            
            # Save the figure
            if save_path:
            try:
                plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
                print(f"Saved entity relationship graph to {save_path}")
            except Exception as e:
                print(f"Warning: Could not save visualization to {save_path}: {e}")
                # Try saving with simpler parameters
                try:
                    plt.savefig(save_path, dpi=150)
                    print(f"Saved entity relationship graph to {save_path} (simplified)")
                except Exception as e2:
                    print(f"Error: Failed to save visualization: {e2}")
            
            # Close the figure to free memory
            plt.close(fig)
            
        # Return None for non-interactive mode
        if non_interactive:
            return None
    else:
            return fig
            
    except Exception as e:
        print(f"Error creating entity relationship graph visualization: {e}")
        print("Continuing without visualization...")
        return None