from deepseek_api import DeepSeekAPI
import re
import json
import pandas as pd
import networkx as nx
import concurrent.futures
from tqdm import tqdm
import time
from difflib import SequenceMatcher
import os

class EntityExtractor:
    def __init__(self, verbose=False, model_name="deepseek-chat", use_cache=True, cache_dir="cache", dataset_type="musique"):
        self.deepseek = DeepSeekAPI(model_name=model_name)
        self.verbose = verbose
        self.model_name = model_name
        self.use_cache = use_cache
        self.dataset_type = dataset_type.lower()
        
        # Create dataset-specific cache directory
        self.cache_dir = os.path.join(cache_dir, self.dataset_type)
        
        # Create cache directory if it doesn't exist
        if use_cache and not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # Create document cache directory
        self.doc_cache_dir = os.path.join(self.cache_dir, "documents")
        if use_cache and not os.path.exists(self.doc_cache_dir):
            os.makedirs(self.doc_cache_dir)
        
        # Load existing cache from disk if available
        self.entity_cache = self._load_cache("entity_cache.json")
        self.relationship_cache = self._load_cache("relationship_cache.json")
        
        # Track if cache has been modified
        self.cache_modified = False
        
        # Document cache tracking
        self.doc_cache_modified = set()  # Track which document caches have been modified
    
    def _load_cache(self, filename):
        """Load cache from disk"""
        cache_path = os.path.join(self.cache_dir, filename)
        if self.use_cache and os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading cache from {cache_path}: {e}")
        return {}

    def _save_cache(self, cache, filename):
        """Save cache to disk"""
        if self.use_cache:
            cache_path = os.path.join(self.cache_dir, filename)
            try:
                with open(cache_path, 'w') as f:
                    json.dump(cache, f, indent=2)
                if self.verbose:
                    print(f"Saved cache to {cache_path}")
            except Exception as e:
                print(f"Error saving cache to {cache_path}: {e}")

    def save_caches(self):
        """Save all caches to disk"""
        if self.cache_modified:
            self._save_cache(self.entity_cache, "entity_cache.json")
            self._save_cache(self.relationship_cache, "relationship_cache.json")
            self.cache_modified = False
        
        # No need to save document caches here as they're saved immediately after modification
    
    def extract_entities_from_question(self, question, known_entities=None, question_id=None):
        """
        Extract entities from a question using DeepSeek API with caching
        
        Args:
            question: The question text
            known_entities: Optional list of previously known entities to consider
            question_id: Optional question ID for more predictable caching
            
        Returns:
            List of entities found in the question
        """
        # Create a cache key using question_id if provided (more predictable)
        if question_id is not None:
            if known_entities:
                cache_key = f"question_{question_id}_with_context"
            else:
                cache_key = f"question_{question_id}"
        else:
            # Fall back to hash-based keys if no question_id provided
            if known_entities:
                # Sort and convert to tuple for consistent hashing
                known_entities_tuple = tuple(sorted(known_entities))
                cache_key = f"question_{hash(question)}_{hash(known_entities_tuple)}"
            else:
                cache_key = f"question_{hash(question)}"
        
        # Check cache first if caching is enabled
        if self.use_cache and cache_key in self.entity_cache:
            if self.verbose:
                print(f"Using cached question entities for key {cache_key}")
            return self.entity_cache[cache_key]
        
        # If we have known entities, use them in the prompt
        if known_entities and len(known_entities) > 0:
            prompt = f"""
            Extract all important entities and concepts from the following question by selecting from the provided list that have already been identified in related texts.
            
            Question: {question}
            
            Previously identified entities and concepts:
            {json.dumps(known_entities, indent=2)}
            
            INSTRUCTIONS:
            1. FIRST, identify which entities or concepts from the provided list appear in the question or are directly relevant to the question.
            2. Consider ALL types of entities and concepts, not just named entities (people, places, organizations).
            3. Include abstract concepts, events, ideas, and other non-named entities that are important to understanding the question.
            4. ONLY select entities from the provided list that are explicitly mentioned or clearly implied in the question.
            5. If NO entities from the list are mentioned in the question, select the 1-3 most relevant entities from the list that would help answer this question.
            6. DO NOT invent new entities that aren't in the provided list.
            
            Return only a JSON array of entity names from the provided list, with no additional text.
            Example: ["Entity1", "Entity2", "Entity3"]
            """
        else:
            # Original implementation for when no known entities are provided
            prompt = f"""
            Extract all important entities and concepts from the following question:
            
            Question: {question}
            
            INSTRUCTIONS:
            1. Extract ALL types of entities and concepts that are important to understanding and answering the question.
            2. Include both named entities (people, places, organizations, products) AND non-named entities (abstract concepts, events, ideas, etc.).
            3. Focus on entities that would be useful for retrieving relevant information to answer the question.
            4. Be comprehensive - don't miss important concepts even if they're not traditional named entities.
            
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
            
            # If we have known entities, ensure all returned entities are in the known list
            if known_entities and len(known_entities) > 0:
                known_entities_set = set(known_entities)
                entities = [entity for entity in entities if entity in known_entities_set]
                
                # If no entities were found in the known list, select the most relevant ones
                if not entities and known_entities:
                    # Use a simple heuristic - select entities that have words in common with the question
                    question_words = set(question.lower().split())
                    entity_scores = []
                    
                    for entity in known_entities:
                        entity_words = set(entity.lower().split())
                        common_words = question_words.intersection(entity_words)
                        score = len(common_words)
                        entity_scores.append((entity, score))
                    
                    # Sort by score (descending) and take top 3
                    entity_scores.sort(key=lambda x: x[1], reverse=True)
                    entities = [entity for entity, score in entity_scores[:3] if score > 0]
                    
                    if self.verbose:
                        print(f"No entities found in question, selected most relevant: {entities}")
            
            # Cache the result if caching is enabled
            if self.use_cache:
                self.entity_cache[cache_key] = entities
                self.cache_modified = True
                
                # Also save immediately to ensure it's persisted
                self._save_cache(self.entity_cache, "entity_cache.json")
            
            return entities
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Raw response: {response}")
            return []
    
    def extract_entities_from_paragraph(self, paragraph_text, doc_id=None):
        """
        Extract entities from a paragraph using DeepSeek API with caching
        
        Args:
            paragraph_text: The paragraph text
            doc_id: Optional document ID for document-specific caching
            
        Returns:
            List of entities found in the paragraph
        """
        # Use doc_id directly as cache key if provided
        if doc_id is not None:
            cache_key = f"paragraph_{doc_id}"
        else:
            # Fall back to hash-based key if no doc_id provided
            cache_key = f"paragraph_{hash(paragraph_text)}"
        
        # Check global cache first if caching is enabled
        if self.use_cache and cache_key in self.entity_cache:
            if self.verbose:
                print(f"Using cached paragraph entities (global cache) for key {cache_key}")
            return self.entity_cache[cache_key]
        
        # Check document-specific cache if doc_id is provided
        if doc_id is not None and self.use_cache:
            doc_cache = self._load_document_cache(doc_id)
            if "entities" in doc_cache:
                if self.verbose:
                    print(f"Using cached paragraph entities for document {doc_id}")
                return doc_cache["entities"]
        
        if self.verbose:
            print(f"No cache found for {doc_id}, extracting entities...")
        
        # Original implementation for entity extraction
        prompt = f"""
        Extract all important entities and concepts from the following paragraph:
        
        Paragraph: {paragraph_text}
        
        INSTRUCTIONS:
        1. Extract ALL types of entities and concepts that are important to understanding the paragraph.
        2. Include both named entities (people, places, organizations, products) AND non-named entities (abstract concepts, events, ideas, etc.).
        3. Focus on entities that would be useful for answering questions about this paragraph.
        4. Be comprehensive - don't miss important concepts even if they're not traditional named entities.
        5. Include key terms, technical concepts, and domain-specific terminology.
        
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
            # Find JSON in the response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
                
            # Clean up the string
            json_str = re.sub(r'```.*?```', '', json_str, flags=re.DOTALL)
            
            # Parse the JSON
            entities = json.loads(json_str)
            
            # Cache the result in global cache
            if self.use_cache:
                self.entity_cache[cache_key] = entities
                self.cache_modified = True
                
                # Also save immediately to ensure it's persisted
                self._save_cache(self.entity_cache, "entity_cache.json")
                if self.verbose:
                    print(f"Saved entities to global cache with key {cache_key}")
            
            # Cache the result in document-specific cache if doc_id is provided
            if doc_id is not None and self.use_cache:
                doc_cache = self._load_document_cache(doc_id)
                doc_cache["entities"] = entities
                self._save_document_cache(doc_id, doc_cache)
                self.doc_cache_modified.add(doc_id)
                if self.verbose:
                    print(f"Saved entities to document cache for {doc_id}")
            
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
        Uses caching to avoid redundant API calls.
        
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
        
        if self.verbose:
            print(f"Initialized entity graph with {len(entities)} entities")
            print(f"Sample entities: {entities[:10] if len(entities) > 10 else entities}")
        
        # Check which documents need relationship extraction
        docs_needing_extraction = []
        cached_relationships = []
        
        for doc_id in reachable_docs:
            doc_text = G.nodes[doc_id]['text']
            doc_entities = [node for node in G.neighbors(doc_id) if G.nodes[node]['type'] == 'entity']
            
            # Only process documents with at least 2 entities
            if len(doc_entities) < 2:
                continue
            
            # Check if relationships are already cached
            # First check document-specific cache
            doc_cache = self._load_document_cache(doc_id)
            if "relationships" in doc_cache:
                if self.verbose:
                    print(f"Using document-specific cached relationships for {doc_id}")
                cached_relationships.append({
                    'doc_id': doc_id,
                    'relationships': doc_cache["relationships"]
                })
                continue
            
            # Then check global cache using the same key format as extract_entity_relationships
            cache_key = f"doc_relationships_{doc_id}"
            if self.use_cache and cache_key in self.relationship_cache:
                if self.verbose:
                    print(f"Using cached relationships for document {doc_id} from global cache")
                cached_relationships.append({
                    'doc_id': doc_id,
                    'relationships': self.relationship_cache[cache_key]
                })
                continue
            
            # Also check the old cache key format from extract_relationships_from_text
            entities_str = ",".join(sorted(doc_entities))
            old_cache_key = f"rel_{hash(doc_text)}_{hash(entities_str)}"
            if self.use_cache and old_cache_key in self.relationship_cache:
                if self.verbose:
                    print(f"Using cached relationships (old format) for document {doc_id}")
                # Convert old format to new format
                old_relationships = self.relationship_cache[old_cache_key]
                new_relationships = []
                for rel in old_relationships:
                    if isinstance(rel, tuple) and len(rel) == 3:
                        new_relationships.append((rel[0], rel[1], rel[2]))
                    
                cached_relationships.append({
                    'doc_id': doc_id,
                    'relationships': new_relationships
                })
                continue
            
            # If not cached, add to extraction list
            docs_needing_extraction.append({
                'doc_id': doc_id,
                'text': doc_text,
                'entities': doc_entities,
                'title': G.nodes[doc_id].get('title', '')
            })
        
        # Process cached relationships first
        for result in cached_relationships:
            doc_id = result['doc_id']
            relationships = result['relationships']
            
            if self.verbose:
                print(f"Processing {len(relationships)} cached relationships for {doc_id}")
            
            for rel in relationships:
                if isinstance(rel, tuple) and len(rel) == 3:
                    # New format: (entity1, entity2, relationship)
                    source, target, relation = rel
                elif isinstance(rel, list) and len(rel) == 3:
                    # List format: [entity1, entity2, relationship]
                    source, target, relation = rel
                elif isinstance(rel, dict) and 'source' in rel and 'target' in rel and 'relation' in rel:
                    # Old format: {"source": ..., "target": ..., "relation": ...}
                    source = rel['source']
                    target = rel['target']
                    relation = rel['relation']
                else:
                    if self.verbose:
                        print(f"Skipping invalid relationship format: {rel}")
                    continue
                
                # Check if both entities exist in the entity graph, add them if missing
                if source not in entity_graph.nodes():
                    if self.verbose:
                        print(f"Adding missing source entity '{source}' to entity graph")
                    entity_graph.add_node(source)
                if target not in entity_graph.nodes():
                    if self.verbose:
                        print(f"Adding missing target entity '{target}' to entity graph")
                    entity_graph.add_node(target)
                
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
                    if self.verbose:
                        print(f"Added relationship: {source} -> {target} ({relation})")
        
        # Process documents that need extraction
        if docs_needing_extraction:
            print(f"Extracting relationships from {len(docs_needing_extraction)} uncached documents...")
            
            # Use the more efficient extract_entity_relationships method
            for doc_data in tqdm(docs_needing_extraction, desc="Extracting relationships"):
                doc_id = doc_data['doc_id']
                doc_text = doc_data['text']
                doc_entities = doc_data['entities']
                doc_title = doc_data['title']
                
                try:
                    # Use extract_entity_relationships which has better caching
                    relationships = self.extract_entity_relationships(
                        doc_id, doc_text, doc_entities, doc_title
                    )
                    
                    # Add relationships to the graph
                    for entity1, entity2, relation in relationships:
                        # Add edge or update if it already exists
                        if entity_graph.has_edge(entity1, entity2):
                            # If the edge exists, append the new relation to the list
                            entity_graph[entity1][entity2]['relations'].append(relation)
                            entity_graph[entity1][entity2]['documents'].add(doc_id)
                            entity_graph[entity1][entity2]['weight'] += 1
                        else:
                            # Create a new edge with the relation
                            entity_graph.add_edge(
                                entity1, entity2, 
                                relations=[relation], 
                                documents={doc_id},
                                weight=1
                            )
                            
                except Exception as e:
                    print(f"Error extracting relationships for document {doc_id}: {e}")
        else:
            print("All relationships found in cache - no API calls needed!")
        
        # Debug: Show graph state before removing isolated nodes
        if self.verbose:
            print(f"Entity graph before cleanup: {entity_graph.number_of_nodes()} nodes, {entity_graph.number_of_edges()} edges")
        
        # Remove isolated nodes (nodes with no edges)
        isolated_nodes = [node for node in entity_graph.nodes() if entity_graph.degree(node) == 0]
        entity_graph.remove_nodes_from(isolated_nodes)
        
        if isolated_nodes:
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
        Extract relationships between entities in a text using DeepSeek API with caching
        
        Args:
            text: The document text
            entities: List of entities to look for relationships between
            
        Returns:
            List of dictionaries with source, target, and relation
        """
        # Create a cache key based on text and entities
        entities_str = ",".join(sorted(entities))
        cache_key = f"rel_{hash(text)}_{hash(entities_str)}"
        
        # Check cache first if caching is enabled
        if self.use_cache and cache_key in self.relationship_cache:
            if self.verbose:
                print("Using cached relationships")
            return self.relationship_cache[cache_key]
        
        # Create a prompt for relationship extraction
        prompt = f"""
        Extract relationships between the following entities in the text:
        
        Entities: {json.dumps(entities)}
        
        Text: {text}
        
        For each relationship, identify:
        1. The source entity
        2. The target entity
        3. The relationship between them (a short phrase or verb)
        
        Return the results as a JSON array of objects with 'source', 'target', and 'relation' fields.
        Example: [
            {{"source": "Entity1", "target": "Entity2", "relation": "works for"}},
            {{"source": "Entity3", "target": "Entity4", "relation": "is located in"}}
        ]
        
        Only include relationships that are explicitly mentioned in the text.
        Only include entities from the provided list.
        """
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that extracts relationships between entities."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.deepseek.generate_response(messages, temperature=0.1)
        
        # Extract JSON from response
        try:
            # Find JSON in the response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
                
            # Clean up the string
            json_str = re.sub(r'```.*?```', '', json_str, flags=re.DOTALL)
            json_str = re.sub(r'```json', '', json_str)
            json_str = re.sub(r'```', '', json_str)
            
            # Parse the JSON
            relationships = json.loads(json_str)
            
            # Validate relationships
            valid_relationships = []
            for rel in relationships:
                if 'source' in rel and 'target' in rel and 'relation' in rel:
                    # Check that source and target are in the entities list
                    if rel['source'] in entities and rel['target'] in entities:
                        valid_relationships.append(rel)
            
            # Cache the result if caching is enabled
            if self.use_cache:
                self.relationship_cache[cache_key] = valid_relationships
                self.cache_modified = True
                
                # Save immediately to ensure it's persisted
                self._save_cache(self.relationship_cache, "relationship_cache.json")
                if self.verbose:
                    print(f"Saved relationships to global cache with key {cache_key}")
            
            return valid_relationships
        except Exception as e:
            print(f"Error parsing relationships: {e}")
            print(f"Raw response: {response}")
            return []  # Return empty list on error

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

    def create_entity_document_graph_experiment(self, example, experiment_type="standard", max_workers=5):
        """Create a bipartite graph with detailed timing and shared caching"""
        print(f"Running experiment: {experiment_type}")
        
        # Extract entities from the question (same for all experiments)
        print("Extracting entities from question...")
        question_start = time.time()
        question_entities = self.extract_entities_from_question(example['question'])
        if self.verbose:
            print(f"Question entity extraction took {time.time() - question_start:.2f} seconds")
        print(f"Entities in question: {question_entities}")
        
        # Create a bipartite graph
        G = nx.Graph()
        
        # Add question entities as nodes
        for entity in question_entities:
            G.add_node(entity, type='entity')
        
        # Pre-extract all paragraph entities to share across experiments
        if experiment_type in ["standard", "fuzzy_matching", "llm_merging"]:
            # These experiments all use the same entity extraction, so pre-extract once
            paragraph_entities = {}
            
            print(f"Pre-extracting entities from {len(example['paragraphs'])} paragraphs...")
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Create a list to store futures
                futures = []
                
                # Submit tasks for each paragraph
                for i, paragraph in enumerate(example['paragraphs']):
                    futures.append(
                        executor.submit(
                            self.extract_entities_from_paragraph,
                            paragraph['paragraph_text']
                        )
                    )
                
                # Process results as they complete
                for i, future in enumerate(tqdm(concurrent.futures.as_completed(futures), 
                                              total=len(futures),
                                              desc="Pre-extracting entities")):
                    try:
                        # Get the paragraph and its entities
                        paragraph = example['paragraphs'][i]
                        entities = future.result()
                        paragraph_entities[i] = entities
                        
                    except Exception as e:
                        print(f"Error pre-extracting paragraph {i}: {e}")
            
            # Now choose the appropriate experiment implementation with pre-extracted entities
            if experiment_type == "fuzzy_matching":
                return self._experiment_fuzzy_matching(G, example, question_entities, paragraph_entities)
            elif experiment_type == "llm_merging":
                return self._experiment_llm_merging(G, example, question_entities, paragraph_entities)
            else:  # standard approach
                return self._experiment_standard(G, example, question_entities, paragraph_entities)
        else:
            # Sequential context needs to process paragraphs in order
            return self._experiment_sequential_context(G, example, question_entities)

    def _experiment_standard(self, G, example, question_entities, paragraph_entities):
        """
        Standard approach: use pre-extracted entities
        """
        print(f"Processing {len(example['paragraphs'])} paragraphs with pre-extracted entities...")
        
        # Process each paragraph with pre-extracted entities
        for i, paragraph in enumerate(tqdm(example['paragraphs'], desc="Processing paragraphs")):
            try:
                # Get the pre-extracted entities
                entities = paragraph_entities[i]
                
                # Create document node
                doc_id = f"doc_{i}"
                G.add_node(doc_id, 
                          type='document', 
                          title=paragraph['title'], 
                          text=paragraph['paragraph_text'],
                          is_supporting=paragraph.get('is_supporting', False))
                
                # Connect entities to document
                for entity in entities:
                    G.add_node(entity, type='entity')
                    G.add_edge(entity, doc_id)
                    
            except Exception as e:
                print(f"Error processing paragraph {i}: {e}")
        
        return G, question_entities

    def _experiment_fuzzy_matching(self, G, example, question_entities, paragraph_entities):
        """
        Fuzzy matching approach: use pre-extracted entities and fuzzy string matching to merge similar entities
        """
        print(f"Processing {len(example['paragraphs'])} paragraphs with fuzzy matching...")
        
        # Collect all entities for fuzzy matching
        all_entities = set(question_entities)
        for entities in paragraph_entities.values():
            all_entities.update(entities)
        
        # Create entity mapping using fuzzy matching
        entity_mapping = self._merge_entities_with_fuzzy_matching(list(all_entities))
        
        # Process each paragraph with pre-extracted entities and apply mapping
        for i, paragraph in enumerate(tqdm(example['paragraphs'], desc="Processing paragraphs")):
            try:
                # Get the pre-extracted entities and map them
                entities = paragraph_entities[i]
                mapped_entities = [entity_mapping.get(entity, entity) for entity in entities]
                
                # Create document node
                doc_id = f"doc_{i}"
                G.add_node(doc_id, 
                          type='document', 
                          title=paragraph['title'], 
                          text=paragraph['paragraph_text'],
                          is_supporting=paragraph.get('is_supporting', False))
                
                # Connect mapped entities to document
                for entity in mapped_entities:
                    G.add_node(entity, type='entity')
                    G.add_edge(entity, doc_id)
                    
            except Exception as e:
                print(f"Error processing paragraph {i}: {e}")
        
        # Map question entities
        mapped_question_entities = [entity_mapping.get(entity, entity) for entity in question_entities]
        
        return G, mapped_question_entities

    def _experiment_llm_merging(self, G, example, question_entities, paragraph_entities=None):
        """
        LLM merging approach: use pre-extracted entities and LLM to merge equivalent entities
        """
        print(f"Processing {len(example['paragraphs'])} paragraphs with LLM merging...")
        
        # Extract entities if not provided
        if paragraph_entities is None:
            paragraph_entities = {}
            for i, paragraph in enumerate(tqdm(example['paragraphs'], desc="Extracting entities")):
                doc_id = f"doc_{i}"
                entities = self.extract_entities_from_paragraph(
                    paragraph['paragraph_text'],
                    doc_id=doc_id
                )
                paragraph_entities[i] = entities
        
        # Collect all entities for LLM merging
        all_entities = set(question_entities)
        for entities in paragraph_entities.values():
            all_entities.update(entities)
        
        # Use LLM to merge entities
        entity_mapping = self.merge_equivalent_entities_with_llm(list(all_entities))
        
        # Process each paragraph with pre-extracted entities and apply mapping
        for i, paragraph in enumerate(tqdm(example['paragraphs'], desc="Processing paragraphs")):
            try:
                # Get the pre-extracted entities and map them
                entities = paragraph_entities[i]
                mapped_entities = [entity_mapping.get(entity, entity) for entity in entities]
                
                # Create document node
                doc_id = f"doc_{i}"
                G.add_node(doc_id, 
                          type='document', 
                          title=paragraph['title'], 
                          text=paragraph['paragraph_text'],
                          is_supporting=paragraph.get('is_supporting', False))
                
                # Connect mapped entities to document
                for entity in mapped_entities:
                    G.add_node(entity, type='entity')
                    G.add_edge(entity, doc_id)
                    
            except Exception as e:
                print(f"Error processing paragraph {i}: {e}")
        
        # Map question entities
        mapped_question_entities = [entity_mapping.get(entity, entity) for entity in question_entities]
        
        return G, mapped_question_entities

    def merge_equivalent_entities_with_llm(self, entities):
        """
        Use LLM to identify and merge equivalent entities
        
        Args:
            entities: List of entities to check for equivalence
            
        Returns:
            Dictionary mapping original entities to canonical forms
        """
        # Skip if there are too few entities
        if len(entities) < 2:
            return {}
        
        # Create a more precise prompt with clear instructions and examples
        prompt = f"""
        I have extracted the following entities from a set of documents:
        {json.dumps(entities, indent=2)}
        
        I need to identify entities that refer to the exact same real-world object, person, or concept but are written differently.
        
        IMPORTANT GUIDELINES:
        1. ONLY merge entities that are truly the same entity with different names/spellings
        2. DO NOT merge entities that are merely related or in the same category
        3. DO NOT merge specific entities into broader categories
        4. DO NOT merge people with organizations they belong to
        5. DO NOT merge movies/books with their creators or characters
        6. Maintain the most specific and accurate form as the canonical entity
        
        Examples of correct merging:
        - "NYC" → "New York City" (different names for the same city)
        - "Barack Obama" → "President Obama" (same person, different references)
        - "IBM" → "International Business Machines" (same company, full vs. acronym)
        
        Examples of INCORRECT merging:
        - "Jennifer Garner" → "Walt Disney Pictures" (an actress is not a studio)
        - "Green Party" → "Citizens Party" (different political parties)
        - "Grant Green" → "Green Album" (a person is not an album)
        
        Return your answer as a JSON object where:
        - Keys are the original entities
        - Values are the canonical forms (choose the most complete/accurate form)
        
        Only include entities that should be merged. If an entity has no equivalent, don't include it.
        """
        
        messages = [
            {"role": "system", "content": "You are a precise entity resolution specialist. Your task is to identify when two differently written entities refer to exactly the same real-world entity. Be extremely conservative - only merge entities when you are certain they are the same."},
            {"role": "user", "content": prompt}
        ]
        
        print("Using LLM to merge equivalent entities...")
        # Use a low temperature for more precise, deterministic results
        response = self.deepseek.generate_response(messages, temperature=0.1)
        
        # Extract JSON from response
        try:
            # Find JSON in the response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
                
            # Clean up the string
            json_str = re.sub(r'```.*?```', '', json_str, flags=re.DOTALL)
            json_str = re.sub(r'```json', '', json_str)
            json_str = re.sub(r'```', '', json_str)
            
            # Parse the JSON
            entity_mapping = json.loads(json_str)
            
            # Validate the mappings - ensure we're not doing crazy merges
            validated_mapping = {}
            for original, canonical in entity_mapping.items():
                # Skip if the original and canonical are the same
                if original == canonical:
                    continue
                    
                # Skip if the similarity is too low (likely incorrect merge)
                similarity = SequenceMatcher(None, original.lower(), canonical.lower()).ratio()
                if similarity < 0.3:  # Threshold for minimum similarity
                    print(f"  Rejected mapping: '{original}' → '{canonical}' (similarity: {similarity:.2f})")
                    continue
                    
                # Accept the mapping
                validated_mapping[original] = canonical
            
            # Print the mappings
            if validated_mapping:
                print("Entity mappings:")
                for original, canonical in validated_mapping.items():
                    print(f"  '{original}' → '{canonical}'")
            else:
                print("No valid entity mappings found.")
            
            return validated_mapping
        except Exception as e:
            print(f"Error parsing entity mapping: {e}")
            print(f"Raw response: {response}")
            return {}  # Return empty mapping on error

    def _experiment_sequential_context(self, G, example, question_entities):
        """
        Sequential context approach: extracts entities sequentially with context from previous extractions
        """
        print(f"Processing {len(example['paragraphs'])} paragraphs sequentially with context...")
        
        # Start with question entities as the initial context
        known_entities = set(question_entities)
        
        # Process paragraphs sequentially
        for i, paragraph in enumerate(tqdm(example['paragraphs'], desc="Extracting entities sequentially")):
            doc_id = f"doc_{i}"
            
            # Extract entities with context
            entities = self._extract_entities_with_context(
                paragraph['paragraph_text'], 
                list(known_entities)
            )
            
            # Update known entities
            known_entities.update(entities)
            
            # Add document node
            G.add_node(doc_id, 
                      type='document', 
                      title=paragraph['title'], 
                      text=paragraph['paragraph_text'],
                      is_supporting=paragraph.get('is_supporting', False))
            
            # Print entities found in this paragraph
            print(f"Paragraph {i} ({paragraph['title']}): {entities}")
            
            # Connect entities to document
            for entity in entities:
                G.add_node(entity, type='entity')
                G.add_edge(entity, doc_id)
        
        return G, question_entities

    def _extract_entities_with_context(self, paragraph_text, known_entities, doc_id=None, ner=True):
        """
        Extract entities from a paragraph with context from previously known entities
        
        Args:
            paragraph_text: The paragraph text
            known_entities: List of previously known entities
            doc_id: Optional document ID for document-specific caching
            
        Returns:
            List of entities found in the paragraph
        """
        # Use doc_id directly as cache key if provided
        if doc_id is not None:
            cache_key = f"paragraph_context_{doc_id}"
        else:
            # Fall back to hash-based key if no doc_id provided
            known_entities_tuple = tuple(sorted(known_entities))
            cache_key = f"paragraph_context_{hash(paragraph_text)}_{hash(known_entities_tuple)}"
        
        # Check global cache first if caching is enabled
        if self.use_cache and cache_key in self.entity_cache:
            if self.verbose:
                print(f"Using cached paragraph entities with context (global cache) for key {cache_key}")
            return self.entity_cache[cache_key]
        
        # Check document-specific cache if doc_id is provided
        if doc_id is not None and self.use_cache:
            doc_cache = self._load_document_cache(doc_id)
            if "entities_with_context" in doc_cache:
                if self.verbose:
                    print(f"Using cached paragraph entities with context for document {doc_id}")
                return doc_cache["entities_with_context"]
        
        if self.verbose:
            print(f"No context cache found for {doc_id}, extracting entities with context...")
            print(f"Context includes {len(known_entities)} known entities")
        
        # Original implementation
        
        if ner:
            prompt = f"""
            Extract all entities from the following paragraph:
            
            Paragraph: {paragraph_text}
            
            An entity is a real-world object such as a person, location, organization, product, etc.
            
            Here are some entities that have already been identified in related texts:
            {json.dumps(known_entities, indent=2)}
            
            Please identify:
            1. Any of the above entities that appear in this paragraph
            2. New entities that haven't been identified yet
            
            Return only a JSON array of entity names, with no additional text.
            Example: ["Entity1", "Entity2", "Entity3"]
            """
        else:
            prompt = f"""
            Extract all important entities and concepts from the following paragraph, considering the list that have already been identified in related texts.
            
            Paragraph: {paragraph_text}
            
            Previously identified entities and concepts:
            {json.dumps(known_entities, indent=2)}
            
            INSTRUCTIONS:
            1. Extract ALL types of entities and concepts that are important to understanding the paragraph.
            2. Include both named entities (people, places, organizations, products) AND non-named entities (abstract concepts, events, ideas, etc.).
            3. First identify any of the previously identified entities that appear in this paragraph.
            4. Then identify new entities and concepts that haven't been identified yet.
            5. Focus on entities that would be useful for answering questions about this paragraph.
            6. Be comprehensive - don't miss important concepts even if they're not traditional named entities.
            
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
            # Find JSON in the response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
                
            # Clean up the string
            json_str = re.sub(r'```.*?```', '', json_str, flags=re.DOTALL)
            
            # Parse the JSON
            entities = json.loads(json_str)
            
            # Cache the result in global cache
            if self.use_cache:
                self.entity_cache[cache_key] = entities
                self.cache_modified = True
            
            # Cache the result in document-specific cache if doc_id is provided
            if doc_id is not None and self.use_cache:
                doc_cache = self._load_document_cache(doc_id)
                doc_cache["entities_with_context"] = entities
                self._save_document_cache(doc_id, doc_cache)
                self.doc_cache_modified.add(doc_id)
            
            if self.verbose:
                print(f"Saved context entities to global cache with key {cache_key}")
                print(f"Found {len(entities)} entities in paragraph with context")
            
            return entities
        except Exception as e:
            print(f"Error parsing response: {e}")
            print(f"Raw response: {response}")
            return []

    def _merge_entities_with_fuzzy_matching(self, entities, threshold=0.85):
        """
        Use fuzzy string matching to identify and merge equivalent entities
        
        Args:
            entities: List of entities to check for equivalence
            threshold: Similarity threshold for merging (0.0 to 1.0)
            
        Returns:
            Dictionary mapping original entities to canonical forms
        """
        # Skip if there are too few entities
        if len(entities) < 2:
            return {}
        
        # Sort entities by length for better canonical selection
        sorted_entities = sorted(entities, key=len, reverse=True)
        
        # Create mapping dictionary
        entity_mapping = {}
        
        # Compare each entity with all others
        for i, entity1 in enumerate(sorted_entities):
            # Skip if this entity is already mapped to something else
            if entity1 in entity_mapping:
                continue
            
            for entity2 in sorted_entities[i+1:]:
                # Skip if entity2 is already mapped
                if entity2 in entity_mapping:
                    continue
                
                # Calculate similarity
                similarity = SequenceMatcher(None, entity1.lower(), entity2.lower()).ratio()
                
                # If similarity is above threshold, map entity2 to entity1
                if similarity >= threshold:
                    entity_mapping[entity2] = entity1
        
        # Print the mappings
        if entity_mapping:
            print("Entity mappings from fuzzy matching:")
            for original, canonical in entity_mapping.items():
                print(f"  '{original}' → '{canonical}' (similarity: {SequenceMatcher(None, original.lower(), canonical.lower()).ratio():.2f})")
        else:
            print("No entity mappings found with fuzzy matching.")
        
        return entity_mapping

    def apply_standard_experiment(self, G, example, question_entities, paragraph_entities=None):
        """
        Apply the standard experiment to the graph using pre-extracted entities
        
        Args:
            G: The initial graph with question entities
            example: The example data
            question_entities: Entities from the question
            paragraph_entities: Pre-extracted entities from paragraphs (optional)
            
        Returns:
            Tuple of (updated graph, mapped question entities)
        """
        print(f"Processing {len(example['paragraphs'])} paragraphs with standard approach...")
        
        # Process each paragraph with pre-extracted entities
        for i, paragraph in enumerate(tqdm(example['paragraphs'], desc="Processing paragraphs")):
            try:
                # Get the pre-extracted entities if provided, otherwise extract them
                if paragraph_entities and i in paragraph_entities:
                    entities = paragraph_entities[i]
                else:
                    # Create a document ID
                    doc_id = f"doc_{i}"
                    entities = self.extract_entities_from_paragraph(
                        paragraph['paragraph_text'],
                        doc_id=doc_id
                    )
                
                # Create document node
                doc_id = f"doc_{i}"
                G.add_node(doc_id, 
                          type='document', 
                          title=paragraph['title'], 
                          text=paragraph['paragraph_text'],
                          is_supporting=paragraph.get('is_supporting', False))
                
                # Connect entities to document
                for entity in entities:
                    G.add_node(entity, type='entity')
                    G.add_edge(entity, doc_id)
                    
            except Exception as e:
                print(f"Error processing paragraph {i}: {e}")
        
        return G, question_entities

    def apply_fuzzy_matching_experiment(self, G, example, question_entities, paragraph_entities=None):
        """
        Apply the fuzzy matching experiment to the graph using pre-extracted entities
        
        Args:
            G: The initial graph with question entities
            example: The example data
            question_entities: Entities from the question
            paragraph_entities: Pre-extracted entities from paragraphs (optional)
            
        Returns:
            Tuple of (updated graph, mapped question entities)
        """
        print(f"Processing {len(example['paragraphs'])} paragraphs with fuzzy matching...")
        
        # Extract entities if not provided
        if paragraph_entities is None:
            paragraph_entities = {}
            for i, paragraph in enumerate(tqdm(example['paragraphs'], desc="Extracting entities")):
                doc_id = f"doc_{i}"
                entities = self.extract_entities_from_paragraph(
                    paragraph['paragraph_text'],
                    doc_id=doc_id
                )
                paragraph_entities[i] = entities
        
        # Collect all entities for fuzzy matching
        all_entities = set(question_entities)
        for entities in paragraph_entities.values():
            all_entities.update(entities)
        
        # Create entity mapping using fuzzy matching
        entity_mapping = self._merge_entities_with_fuzzy_matching(list(all_entities))
        
        # Process each paragraph with pre-extracted entities and apply mapping
        for i, paragraph in enumerate(tqdm(example['paragraphs'], desc="Processing paragraphs")):
            try:
                # Get the pre-extracted entities and map them
                entities = paragraph_entities[i]
                mapped_entities = [entity_mapping.get(entity, entity) for entity in entities]
                
                # Create document node
                doc_id = f"doc_{i}"
                G.add_node(doc_id, 
                          type='document', 
                          title=paragraph['title'], 
                          text=paragraph['paragraph_text'],
                          is_supporting=paragraph.get('is_supporting', False))
                
                # Connect mapped entities to document
                for entity in mapped_entities:
                    G.add_node(entity, type='entity')
                    G.add_edge(entity, doc_id)
                    
            except Exception as e:
                print(f"Error processing paragraph {i}: {e}")
        
        # Map question entities
        mapped_question_entities = [entity_mapping.get(entity, entity) for entity in question_entities]
        
        return G, mapped_question_entities

    def apply_llm_merging_experiment(self, G, example, question_entities, paragraph_entities=None):
        """
        Apply the LLM merging experiment to the graph using pre-extracted entities
        
        Args:
            G: The initial graph with question entities
            example: The example data
            question_entities: Entities from the question
            paragraph_entities: Pre-extracted entities from paragraphs (optional)
            
        Returns:
            Tuple of (updated graph, mapped question entities)
        """
        print(f"Processing {len(example['paragraphs'])} paragraphs with LLM merging...")
        
        # Extract entities if not provided
        if paragraph_entities is None:
            paragraph_entities = {}
            for i, paragraph in enumerate(tqdm(example['paragraphs'], desc="Extracting entities")):
                doc_id = f"doc_{i}"
                entities = self.extract_entities_from_paragraph(
                    paragraph['paragraph_text'],
                    doc_id=doc_id
                )
                paragraph_entities[i] = entities
        
        # Collect all entities for LLM merging
        all_entities = set(question_entities)
        for entities in paragraph_entities.values():
            all_entities.update(entities)
        
        # Use LLM to merge entities
        entity_mapping = self.merge_equivalent_entities_with_llm(list(all_entities))
        
        # Process each paragraph with pre-extracted entities and apply mapping
        for i, paragraph in enumerate(tqdm(example['paragraphs'], desc="Processing paragraphs")):
            try:
                # Get the pre-extracted entities and map them
                entities = paragraph_entities[i]
                mapped_entities = [entity_mapping.get(entity, entity) for entity in entities]
                
                # Create document node
                doc_id = f"doc_{i}"
                G.add_node(doc_id, 
                          type='document', 
                          title=paragraph['title'], 
                          text=paragraph['paragraph_text'],
                          is_supporting=paragraph.get('is_supporting', False))
                
                # Connect mapped entities to document
                for entity in mapped_entities:
                    G.add_node(entity, type='entity')
                    G.add_edge(entity, doc_id)
                    
            except Exception as e:
                print(f"Error processing paragraph {i}: {e}")
        
        # Map question entities
        mapped_question_entities = [entity_mapping.get(entity, entity) for entity in question_entities]
        
        return G, mapped_question_entities

    def apply_sequential_context_experiment(self, G, example, question_entities, example_id=None):
        """
        Apply the sequential context experiment to the graph
        
        Args:
            G: The initial graph with question entities
            example: The example data
            question_entities: Entities from the question
            example_id: Optional example ID for caching
            
        Returns:
            Tuple of (updated graph, question entities)
        """
        print(f"Processing {len(example['paragraphs'])} paragraphs sequentially with context...")
        
        # Start with question entities as the initial context
        known_entities = set(question_entities)
        
        # Process paragraphs sequentially
        for i, paragraph in enumerate(tqdm(example['paragraphs'], desc="Extracting entities sequentially")):
            # Create document ID for caching
            doc_id = f"example_{example_id}_doc_{i}" if example_id is not None else f"doc_{i}"
            
            # Extract entities with context
            entities = self._extract_entities_with_context(
                paragraph['paragraph_text'], 
                list(known_entities),
                doc_id=doc_id
            )
            
            # Update known entities
            known_entities.update(entities)
            
            # Add document node
            G.add_node(doc_id, 
                      type='document', 
                      title=paragraph['title'], 
                      text=paragraph['paragraph_text'],
                      is_supporting=paragraph.get('is_supporting', False))
            
            # Print entities found in this paragraph
            print(f"Paragraph {i} ({paragraph['title']}): {entities}")
            
            # Connect entities to document
            for entity in entities:
                G.add_node(entity, type='entity')
                G.add_edge(entity, doc_id)
            
            # NOTE: We no longer extract relationships here during initial graph construction.
            # Relationships will be extracted later only for reachable documents in 
            # create_entity_relationship_graph method.
        
        return G, question_entities

    def _get_document_cache_path(self, doc_id):
        """Get the cache file path for a specific document"""
        # Create a safe filename from the document ID
        safe_id = str(doc_id).replace('/', '_').replace('\\', '_')
        return os.path.join(self.doc_cache_dir, f"doc_{safe_id}.json")

    def _load_document_cache(self, doc_id):
        """Load cache for a specific document"""
        if not self.use_cache:
            return {}
        
        cache_path = self._get_document_cache_path(doc_id)
        if os.path.exists(cache_path):
            try:
                with open(cache_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                if self.verbose:
                    print(f"Error loading document cache for {doc_id}: {e}")
        return {}

    def _save_document_cache(self, doc_id, cache_data):
        """Save cache for a specific document"""
        if not self.use_cache:
            return
        
        cache_path = self._get_document_cache_path(doc_id)
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
            if self.verbose:
                print(f"Saved document cache to {cache_path}")
        except Exception as e:
            print(f"Error saving document cache for {doc_id}: {e}")

    def check_example_cache(self, example_id, example):
        """
        Check if we have cached data for this example
        
        Args:
            example_id: ID of the example
            example: The example data
            
        Returns:
            Boolean indicating if cached data is available
        """
        if not self.use_cache:
            return False
        
        # Check if question entities are cached (using direct ID)
        question_cache_key = f"question_{example_id}"
        question_cached = question_cache_key in self.entity_cache
        
        # Check if document entities are cached
        doc_cache_available = 0
        total_docs = len(example['paragraphs'])
        for i, paragraph in enumerate(example['paragraphs']):
            doc_id = f"example_{example_id}_doc_{i}"
            doc_cache_path = self._get_document_cache_path(doc_id)
            if os.path.exists(doc_cache_path):
                doc_cache_available += 1
        
        # Check if relationship cache exists for this example
        relationship_cache_available = 0
        for i, paragraph in enumerate(example['paragraphs']):
            doc_id = f"example_{example_id}_doc_{i}"
            relationship_key = f"doc_relationships_{doc_id}"
            if relationship_key in self.relationship_cache:
                relationship_cache_available += 1
        
        # Log cache status
        if self.verbose:
            print(f"Cache status for example {example_id}:")
            print(f"  Question entities cached: {question_cached}")
            print(f"  Document entities cached: {doc_cache_available}/{total_docs} documents")
            print(f"  Relationships cached: {relationship_cache_available}/{total_docs} documents")
        
        # Return True if at least some cache is available
        return question_cached or doc_cache_available > 0 or relationship_cache_available > 0

    def extract_entity_relationships(self, doc_id, doc_text, entities, doc_title=None):
        """
        Extract relationships between entities in a document
        
        Args:
            doc_id: Document ID
            doc_text: Document text
            entities: List of entities to extract relationships for
            doc_title: Optional document title
            
        Returns:
            List of relationship tuples (entity1, entity2, relationship)
        """
        # Skip if there are too few entities
        if len(entities) < 2:
            if self.verbose:
                print(f"Skipping relationship extraction for {doc_id} - only {len(entities)} entities found")
            return []
        
        # Create a cache key using doc_id directly
        cache_key = f"doc_relationships_{doc_id}"
        
        # Check global cache first if caching is enabled
        if self.use_cache and cache_key in self.relationship_cache:
            if self.verbose:
                print(f"Using cached relationships for document {doc_id} from global cache")
            return self.relationship_cache[cache_key]
        
        # Check document-specific cache if doc_id is provided
        if self.use_cache:
            doc_cache = self._load_document_cache(doc_id)
            if "relationships" in doc_cache:
                if self.verbose:
                    print(f"Using document-specific cached relationships for {doc_id}")
                return doc_cache["relationships"]
        
        if self.verbose:
            print(f"Extracting relationships for {doc_id} with {len(entities)} entities")
        
        # Original implementation
        title_info = f" titled '{doc_title}'" if doc_title else ""
        
        prompt = f"""
        Extract relationships between entities in the following document{title_info}:
        
        Document: {doc_text}
        
        Entities: {json.dumps(entities)}
        
        For each pair of entities that have a relationship, describe the relationship in a concise phrase.
        Return the results as a JSON array of objects with the following format:
        [
          {{
            "entity1": "Entity1",
            "entity2": "Entity2",
            "relationship": "relationship description"
          }},
          ...
        ]
        
        Only include relationships that are explicitly mentioned or can be directly inferred from the document.
        If no relationships are found, return an empty array.
        """
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that extracts entity relationships from text."},
            {"role": "user", "content": prompt}
        ]
        
        response = self.deepseek.generate_response(messages, temperature=0.1)
        
        # Extract JSON from response
        try:
            # Find JSON in the response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response
                
            # Clean up the string
            json_str = re.sub(r'```.*?```', '', json_str, flags=re.DOTALL)
            
            # Parse the JSON
            relationships_data = json.loads(json_str)
            
            # Convert to list of tuples
            relationships = []
            for item in relationships_data:
                if 'entity1' in item and 'entity2' in item and 'relationship' in item:
                    relationships.append((item['entity1'], item['entity2'], item['relationship']))
            
            if self.verbose:
                print(f"Found {len(relationships)} relationships for {doc_id}")
            
            # Cache the result in global cache
            if self.use_cache:
                self.relationship_cache[cache_key] = relationships
                self.cache_modified = True
                
                # Save immediately to ensure it's persisted
                self._save_cache(self.relationship_cache, "relationship_cache.json")
                if self.verbose:
                    print(f"Saved relationships to global cache with key {cache_key}")
            
            # Cache the result in document-specific cache
            if self.use_cache:
                doc_cache = self._load_document_cache(doc_id)
                doc_cache["relationships"] = relationships
                self._save_document_cache(doc_id, doc_cache)
                self.doc_cache_modified.add(doc_id)
                if self.verbose:
                    print(f"Saved relationships to document cache for {doc_id}")
            
            return relationships
        except Exception as e:
            print(f"Error parsing relationships: {e}")
            print(f"Raw response: {response}")
            return []

    def generate_answer_baseline(self, question, doc_ids, G):
        """
        Generate an answer to the question using all documents directly without entity extraction
        
        Args:
            question: The question to answer
            doc_ids: List of document IDs to use
            G: The original bipartite graph containing document nodes
            
        Returns:
            The generated answer
        """
        # Collect text from all documents
        doc_texts = []
        for doc_id in doc_ids:
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