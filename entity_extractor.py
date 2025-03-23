from deepseek_api import DeepSeekAPI
import re
import json
import pandas as pd
import networkx as nx
import concurrent.futures
from tqdm import tqdm

class EntityExtractor:
    def __init__(self):
        self.deepseek = DeepSeekAPI()
    
    def extract_entities_from_question(self, question):
        """Extract entities from a question using DeepSeek API"""
        prompt = f"""
        Extract all entities from the following question:
        
        Question: {question}
        
        An entity is a real-world object such as a person, location, organization, product, etc.
        Return only a JSON array of entity names, with no additional text.
        Example: ["Entity1", "Entity2", "Entity3"]
        """
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that extracts entities from text."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.deepseek.generate_response(messages, temperature=0.1)
        
        # Extract JSON from response
        try:
            # Find JSON in the response using regex
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
                
            # Clean up the string to ensure it's valid JSON
            json_str = re.sub(r'```.*?```', '', json_str, flags=re.DOTALL)
            
            # Parse the JSON
            entities = json.loads(json_str)
            return entities
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Raw response: {response}")
            return []
    
    def extract_entities_from_paragraph(self, paragraph_text):
        """Extract entities from a paragraph using DeepSeek API"""
        prompt = f"""
        Extract all entities from the following paragraph:
        
        Paragraph: {paragraph_text}
        
        An entity is a real-world object such as a person, location, organization, product, etc.
        Return only a JSON array of entity names, with no additional text.
        Example: ["Entity1", "Entity2", "Entity3"]
        """
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that extracts entities from text."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.deepseek.generate_response(messages, temperature=0.1)
        
        # Extract JSON from response
        try:
            # Find JSON in the response using regex
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
                
            # Clean up the string to ensure it's valid JSON
            json_str = re.sub(r'```.*?```', '', json_str, flags=re.DOTALL)
            
            # Parse the JSON
            entities = json.loads(json_str)
            return entities
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Raw response: {response}")
            return []
    
    def process_paragraph(self, paragraph, idx):
        """Process a single paragraph and return its entities"""
        paragraph_entities = self.extract_entities_from_paragraph(paragraph['paragraph_text'])
        return {
            'idx': idx,
            'title': paragraph['title'],
            'entities': paragraph_entities,
            'is_supporting': paragraph.get('is_supporting', False)
        }
    
    def create_entity_document_graph(self, example, max_workers=5):
        """Create a bipartite graph connecting entities to documents using parallel processing"""
        # Extract entities from the question
        print("Extracting entities from question...")
        question_entities = self.extract_entities_from_question(example['question'])
        print(f"Entities in question: {question_entities}")
        
        # Create a bipartite graph
        G = nx.Graph()
        
        # Add question entities as nodes
        for entity in question_entities:
            G.add_node(entity, type='entity')
        
        # Process paragraphs in parallel
        print(f"Processing {len(example['paragraphs'])} paragraphs in parallel...")
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all paragraph processing tasks
            future_to_idx = {
                executor.submit(self.process_paragraph, paragraph, i): i 
                for i, paragraph in enumerate(example['paragraphs'])
            }
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_idx), 
                              total=len(future_to_idx),
                              desc="Extracting entities from paragraphs"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing paragraph: {e}")
        
        # Sort results by original index
        results.sort(key=lambda x: x['idx'])
        
        # Add document nodes and connect entities
        for result in results:
            idx = result['idx']
            doc_id = f"doc_{idx}"
            
            # Add document node
            G.add_node(doc_id, 
                      type='document', 
                      title=result['title'], 
                      text=example['paragraphs'][idx]['paragraph_text'],
                      is_supporting=result['is_supporting'])
            
            # Print entities found in this paragraph
            print(f"Paragraph {idx} ({result['title']}): {result['entities']}")
            
            # Connect entities to document
            for entity in result['entities']:
                G.add_node(entity, type='entity')
                G.add_edge(entity, doc_id)
        
        return G, question_entities
    
    def find_reachable_documents(self, G, question_entities, max_hops=2):
        """Find documents reachable from question entities within max_hops"""
        reachable_docs = set()
        
        for entity in question_entities:
            if entity in G:
                # For each entity in the question, find reachable documents
                for node in nx.single_source_shortest_path_length(G, entity, cutoff=max_hops):
                    if isinstance(node, str) and node.startswith('doc_'):
                        reachable_docs.add(node)
        
        return reachable_docs

    def create_entity_relationship_graph(self, G, reachable_docs):
        """
        Create a relationship graph between entities based on reachable documents.
        Uses DeepSeek API to extract relationships between entities in parallel.
        
        Args:
            G: The bipartite entity-document graph
            reachable_docs: Set of document IDs that are reachable
            
        Returns:
            A new graph where entities are connected with their relationships
        """
        # Create a new directed graph for entity relationships
        entity_graph = nx.DiGraph()
        
        # Get all entities from the original graph
        entities = [node for node in G.nodes() if G.nodes[node]['type'] == 'entity']
        
        # Add all entities as nodes
        for entity in entities:
            entity_graph.add_node(entity)
        
        # Process each reachable document
        print(f"Extracting relationships from {len(reachable_docs)} documents in parallel...")
        
        # Prepare document data for parallel processing
        doc_data = []
        for doc_id in reachable_docs:
            doc_text = G.nodes[doc_id]['text']
            doc_entities = [node for node in G.neighbors(doc_id) if G.nodes[node]['type'] == 'entity']
            
            # Only process documents with at least 2 entities
            if len(doc_entities) >= 2:
                doc_data.append({
                    'doc_id': doc_id,
                    'text': doc_text,
                    'entities': doc_entities
                })
        
        # Process documents in parallel
        results = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all document processing tasks
            future_to_doc = {
                executor.submit(self.process_document_relationships, doc): doc 
                for doc in doc_data
            }
            
            # Process results as they complete
            for future in tqdm(concurrent.futures.as_completed(future_to_doc), 
                              total=len(future_to_doc),
                              desc="Extracting relationships"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing document: {e}")
        
        # Add all relationships to the graph
        for result in results:
            doc_id = result['doc_id']
            relationships = result['relationships']
            
            for rel in relationships:
                source = rel['source']
                target = rel['target']
                relation = rel['relation']
                
                # Add edge or update if it already exists
                if entity_graph.has_edge(source, target):
                    # If the edge exists, append the new relation to the list
                    entity_graph[source][target]['relations'].append(relation)
                    entity_graph[source][target]['documents'].add(doc_id)
                    entity_graph[source][target]['weight'] += 1
                else:
                    # Create a new edge with the relation
                    entity_graph.add_edge(
                        source, target, 
                        relations=[relation], 
                        documents={doc_id},
                        weight=1
                    )
        
        # Remove isolated nodes (nodes with no edges)
        isolated_nodes = [node for node in entity_graph.nodes() if entity_graph.degree(node) == 0]
        entity_graph.remove_nodes_from(isolated_nodes)
        
        print(f"Removed {len(isolated_nodes)} isolated nodes from the relationship graph")
        
        return entity_graph

    def process_document_relationships(self, doc_data):
        """
        Process a single document to extract relationships between entities
        
        Args:
            doc_data: Dictionary containing document ID, text, and entities
            
        Returns:
            Dictionary with document ID and extracted relationships
        """
        doc_id = doc_data['doc_id']
        text = doc_data['text']
        entities = doc_data['entities']
        
        # Extract relationships
        relationships = self.extract_relationships_from_text(text, entities)
        
        return {
            'doc_id': doc_id,
            'relationships': relationships
        }

    def extract_relationships_from_text(self, text, entities):
        """
        Extract relationships between entities in a text using DeepSeek API
        
        Args:
            text: The document text
            entities: List of entities to look for relationships between
            
        Returns:
            List of dictionaries with source, target, and relation
        """
        # Truncate text if it's too long
        # if len(text) > 4000:
        #     text = text[:4000] + "..."
        
        # Create a prompt for relationship extraction
        prompt = f"""
        Extract relationships between the following entities in the text:
        
        Entities: {', '.join(entities)}
        
        Text: {text}
        
        For each relationship you find, return a JSON object with the following format:
        {{
            "source": "entity1",
            "target": "entity2",
            "relation": "description of the relationship"
        }}
        
        Return your answer as a JSON array of these relationship objects. Only include relationships that are explicitly mentioned in the text.
        If no relationships are found, return an empty array.
        """
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that extracts relationships between entities from text."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.deepseek.generate_response(messages, temperature=0.1)
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
                
            # Clean up the string to ensure it's valid JSON
            json_str = re.sub(r'```.*?```', '', json_str, flags=re.DOTALL)
            json_str = re.sub(r'```json', '', json_str)
            json_str = re.sub(r'```', '', json_str)
            
            # Parse the JSON
            relationships = json.loads(json_str)
            return relationships
        except Exception as e:
            print(f"Error extracting relationships: {e}")
            print(f"Raw response: {response}")
            return []

    def generate_graph_text_representation(self, entity_graph, question_entities):
        """
        Convert the entity relationship graph into a human-readable text representation
        
        Args:
            entity_graph: The entity relationship graph
            question_entities: List of entities from the question
            
        Returns:
            A string containing a text representation of the graph
        """
        text_parts = ["ENTITY RELATIONSHIP GRAPH:"]
        
        # Add information about question entities
        if question_entities:
            text_parts.append("\nEntities mentioned in the question:")
            for entity in question_entities:
                if entity in entity_graph:
                    text_parts.append(f"- {entity}")
        
        # Add information about relationships
        text_parts.append("\nRelationships between entities:")
        
        # Sort relationships by weight (most important first)
        edge_data = [(u, v, d) for u, v, d in entity_graph.edges(data=True)]
        edge_data.sort(key=lambda x: x[2]['weight'], reverse=True)
        
        for u, v, d in edge_data:
            relations = d['relations']
            weight = d['weight']
            
            # Format the relationships
            if relations:
                # Show the primary relationship and weight
                primary_relation = relations[0]
                text_parts.append(f"- {u} → {v}: {primary_relation} (mentioned {weight} times)")
                
                # If there are multiple relations, show them as well
                if len(relations) > 1:
                    for rel in relations[1:]:
                        text_parts.append(f"  • {u} also {rel} {v}")
        
        return "\n".join(text_parts)

    def generate_answer(self, question, entity_graph, question_entities, reachable_docs, G):
        """
        Generate an answer to the question using the entity graph and reachable documents
        
        Args:
            question: The question to answer
            entity_graph: The entity relationship graph
            question_entities: List of entities from the question
            reachable_docs: Set of document IDs that are reachable
            G: The original bipartite entity-document graph
            
        Returns:
            The generated answer
        """
        # Generate text representation of the graph
        graph_text = self.generate_graph_text_representation(entity_graph, question_entities)
        
        # Collect text from reachable documents
        doc_texts = []
        for doc_id in reachable_docs:
            title = G.nodes[doc_id]['title']
            text = G.nodes[doc_id]['text']
            is_supporting = G.nodes[doc_id].get('is_supporting', False)
            
            # Mark supporting documents
            support_marker = "[SUPPORTING]" if is_supporting else ""
            doc_texts.append(f"DOCUMENT {doc_id} {support_marker}: {title}\n{text}")
        
        # Combine all documents into a single context
        all_docs_text = "\n\n".join(doc_texts)
        
        # Create the prompt for DeepSeek
        prompt = f"""
        I need to answer a question based on the following information:
        
        QUESTION: {question}
        
        {graph_text}
        
        RELEVANT DOCUMENTS:
        {all_docs_text}
        
        Please provide ONLY the exact answer to the question - no explanations, no additional text.
        For example, if the question asks "Who directed Titanic?" just answer "James Cameron".
        Your answer should be as concise as possible, ideally just a name, date, or short phrase.
        """
        
        # Generate the answer using DeepSeek
        messages = [
            {"role": "system", "content": "You are a helpful assistant that provides concise, factual answers based on provided information."},
            {"role": "user", "content": prompt}
        ]
        
        print("Generating answer using DeepSeek API...")
        response = self.deepseek.generate_response(messages, temperature=0.1)
        
        # Clean up the response to remove any explanatory text
        # Look for patterns like "The answer is X" or "X is the answer"
        clean_response = response.strip()
        
        # Remove common prefixes
        prefixes = [
            "The answer is ", "Answer: ", "The correct answer is ", 
            "Based on the information, ", "According to the documents, ",
            "From the information provided, "
        ]
        
        for prefix in prefixes:
            if clean_response.startswith(prefix):
                clean_response = clean_response[len(prefix):]
        
        # Remove quotes if they wrap the entire answer
        if (clean_response.startswith('"') and clean_response.endswith('"')) or \
           (clean_response.startswith("'") and clean_response.endswith("'")):
            clean_response = clean_response[1:-1]
        
        # Remove periods at the end
        if clean_response.endswith('.'):
            clean_response = clean_response[:-1]
        
        return clean_response