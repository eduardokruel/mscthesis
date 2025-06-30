import numpy as np
import re
from typing import List, Dict, Tuple, Set
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from deepseek_api import DeepSeekAPI
import json
import time
import os

# Try to import sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    print("Warning: sentence-transformers not available. Embeddings method will not work.")
    print("Install with: pip install sentence-transformers")

class SemanticDocumentFilter:
    """
    Semantic filtering for documents retrieved through entity-based methods.
    Applies semantic similarity filtering when too many documents are retrieved.
    """
    
    def __init__(self, model_name="deepseek-chat", use_cache=True, cache_dir="cache", verbose=False, 
                 embeddings_model="all-MiniLM-L6-v2"):
        """
        Initialize the semantic filter
        
        Args:
            model_name: Model to use for LLM-based filtering
            use_cache: Whether to use caching for LLM responses
            cache_dir: Directory for caching
            verbose: Whether to print verbose output
            embeddings_model: Model to use for embeddings (if sentence-transformers is available)
        """
        self.deepseek = DeepSeekAPI(model_name=model_name)
        self.use_cache = use_cache
        self.cache_dir = cache_dir
        self.verbose = verbose
        self.embeddings_model_name = embeddings_model
        
        # Initialize TF-IDF vectorizer for semantic similarity
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.95
        )
        
        # Initialize embeddings model if available
        self.embeddings_model = None
        if EMBEDDINGS_AVAILABLE:
            try:
                if self.verbose:
                    print(f"Loading embeddings model: {embeddings_model}")
                self.embeddings_model = SentenceTransformer(embeddings_model)
                if self.verbose:
                    print("Embeddings model loaded successfully")
            except Exception as e:
                print(f"Warning: Failed to load embeddings model: {e}")
                self.embeddings_model = None
        
        # Cache for semantic filtering results
        self.semantic_cache = {}
    
    def predict_question_hops(self, question: str) -> int:
        """
        Predict the number of hops required to answer a question using LLM
        
        Args:
            question: The question to analyze
            
        Returns:
            Predicted number of hops (1, 2, 3, etc.)
        """
        # Create a cache key for this question
        cache_key = f"hop_prediction_{hash(question)}"
        
        # Check cache first
        if self.use_cache and cache_key in self.semantic_cache:
            if self.verbose:
                print(f"Using cached hop prediction: {self.semantic_cache[cache_key]}")
            return self.semantic_cache[cache_key]
        
        # Create prompt for hop prediction
        prompt = f"""
        Analyze the following question and determine how many "hops" or steps of reasoning are required to answer it.
        
        A "hop" represents one step of connecting information across different documents or entities.
        
        Question: {question}
        
        Examples:
        - "Who directed Titanic?" → 1 hop (direct lookup)
        - "What year was the director of Titanic born?" → 2 hops (find director, then find birth year)
        - "What is the birth year of the director of the movie that won Best Picture in 1998?" → 3 hops (find 1998 Best Picture winner, find its director, find director's birth year)
        
        Consider:
        1. How many different pieces of information need to be connected?
        2. How many intermediate lookups are required?
        3. The complexity of the reasoning chain needed
        
        Return only a single integer representing the number of hops (1, 2, 3, etc.).
        """
        
        messages = [
            {"role": "system", "content": "You are an expert at analyzing question complexity and determining the number of reasoning steps required to answer questions."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.deepseek.generate_response(messages, temperature=0.1)
            
            # Extract the number from the response
            import re
            number_match = re.search(r'\b(\d+)\b', response.strip())
            if number_match:
                predicted_hops = int(number_match.group(1))
                # Ensure reasonable bounds (1-5 hops)
                predicted_hops = max(1, min(5, predicted_hops))
            else:
                # Default to 2 hops if parsing fails
                predicted_hops = 2
                if self.verbose:
                    print(f"Could not parse hop prediction from response: '{response}', defaulting to 2")
            
            # Cache the result
            if self.use_cache:
                self.semantic_cache[cache_key] = predicted_hops
            
            if self.verbose:
                print(f"Predicted hops for question: {predicted_hops}")
            
            return predicted_hops
            
        except Exception as e:
            if self.verbose:
                print(f"Error predicting hops: {e}, defaulting to 2")
            return 2  # Default fallback
    
    def calculate_dynamic_threshold(self, question: str, base_threshold: int = 5, 
                                  hop_modifier: int = 0) -> int:
        """
        Calculate a dynamic threshold based on question complexity
        
        Args:
            question: The question to analyze
            base_threshold: Base threshold to start with
            hop_modifier: Modifier to add/subtract from predicted hops
            
        Returns:
            Dynamic threshold value
        """
        predicted_hops = self.predict_question_hops(question)
        adjusted_hops = predicted_hops + hop_modifier
        
        # Calculate dynamic threshold based on hops
        # More hops = more complex = higher threshold needed
        # if adjusted_hops <= 1:
        #     dynamic_threshold = max(3, base_threshold - 2)  # Simple questions, lower threshold
        # elif adjusted_hops == 2:
        #     dynamic_threshold = base_threshold  # Default threshold
        # elif adjusted_hops == 3:
        #     dynamic_threshold = base_threshold + 2  # Complex questions, higher threshold
        # else:  # 4+ hops
        #     dynamic_threshold = base_threshold + 4  # Very complex questions, much higher threshold
        
        if self.verbose:
            print(f"Question complexity analysis:")
            print(f"  Predicted hops: {predicted_hops}")
            print(f"  Hop modifier: {hop_modifier}")
            print(f"  Adjusted hops: {adjusted_hops}")
            print(f"  Base threshold: {base_threshold}")
            # print(f"  Dynamic threshold: {dynamic_threshold}")
        
        return adjusted_hops
    
    def filter_documents_if_needed(self, question: str, reachable_docs: List[str], G: nx.Graph, 
                                 threshold: int = 5, target_docs: int = None, 
                                 method: str = "hybrid", use_dynamic_threshold: bool = False,
                                 hop_modifier: int = 0) -> Tuple[List[str], Dict]:
        """
        Apply semantic filtering if more than threshold documents are retrieved
        
        Args:
            question: The question being answered
            reachable_docs: List of document IDs from entity-based retrieval
            G: NetworkX graph containing document nodes with text
            threshold: Minimum number of docs to trigger filtering (default: 5)
            target_docs: Target number of documents to keep (default: threshold)
            method: Filtering method ("tfidf", "llm", "embeddings", "hybrid")
            use_dynamic_threshold: Whether to use dynamic threshold based on question complexity
            hop_modifier: Modifier to add/subtract from predicted hops for dynamic threshold
            
        Returns:
            Tuple of (filtered_docs, filtering_info)
        """
        # Calculate dynamic threshold if requested
        if use_dynamic_threshold:
            original_threshold = threshold
            threshold = self.calculate_dynamic_threshold(question, threshold, hop_modifier)
            if self.verbose:
                print(f"Using dynamic threshold: {original_threshold} → {threshold}")
        
        if target_docs is None:
            target_docs = threshold
        
        filtering_info = {
            "original_count": len(reachable_docs),
            "threshold": threshold,
            "filtering_applied": False,
            "method_used": None,
            "final_count": len(reachable_docs),
            "filtering_time": 0.0,
            "removed_docs": [],
            "kept_docs": reachable_docs.copy(),
            "dynamic_threshold_used": use_dynamic_threshold,
            "hop_modifier": hop_modifier if use_dynamic_threshold else None
        }
        
        # Check if filtering is needed
        if len(reachable_docs) <= threshold:
            if self.verbose:
                print(f"No semantic filtering needed: {len(reachable_docs)} docs <= {threshold} threshold")
            return reachable_docs, filtering_info
        
        if self.verbose:
            print(f"Applying semantic filtering: {len(reachable_docs)} docs > {threshold} threshold")
            print(f"Target: reduce to {target_docs} documents using {method} method")
        
        start_time = time.time()
        
        # Apply the selected filtering method
        if method == "tfidf":
            filtered_docs = self._filter_with_tfidf(question, reachable_docs, G, target_docs)
        elif method == "llm":
            filtered_docs = self._filter_with_llm(question, reachable_docs, G, target_docs)
        elif method == "embeddings":
            filtered_docs = self._filter_with_embeddings(question, reachable_docs, G, target_docs)
        elif method == "hybrid":
            filtered_docs = self._filter_hybrid(question, reachable_docs, G, target_docs)
        else:
            raise ValueError(f"Unknown filtering method: {method}")
        
        end_time = time.time()
        
        # Update filtering info
        filtering_info.update({
            "filtering_applied": True,
            "method_used": method,
            "final_count": len(filtered_docs),
            "filtering_time": end_time - start_time,
            "removed_docs": [doc for doc in reachable_docs if doc not in filtered_docs],
            "kept_docs": filtered_docs
        })
        
        if self.verbose:
            print(f"Semantic filtering completed in {filtering_info['filtering_time']:.2f}s")
            print(f"Reduced from {filtering_info['original_count']} to {filtering_info['final_count']} documents")
        
        return filtered_docs, filtering_info
    
    def _filter_with_tfidf(self, question: str, doc_ids: List[str], G: nx.Graph, 
                          target_docs: int) -> List[str]:
        """
        Filter documents using TF-IDF cosine similarity
        
        Args:
            question: The question text
            doc_ids: List of document IDs to filter
            G: NetworkX graph containing document nodes
            target_docs: Number of documents to keep
            
        Returns:
            List of filtered document IDs
        """
        # Extract document texts
        doc_texts = []
        valid_doc_ids = []
        
        for doc_id in doc_ids:
            if doc_id in G.nodes and 'text' in G.nodes[doc_id]:
                doc_text = G.nodes[doc_id]['text']
                title = G.nodes[doc_id].get('title', '')
                # Combine title and text for better matching
                combined_text = f"{title} {doc_text}" if title else doc_text
                doc_texts.append(combined_text)
                valid_doc_ids.append(doc_id)
        
        if len(doc_texts) <= target_docs:
            return valid_doc_ids
        
        # Create TF-IDF vectors
        all_texts = [question] + doc_texts
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        
        # Calculate cosine similarity between question and each document
        question_vector = tfidf_matrix[0:1]  # First row is the question
        doc_vectors = tfidf_matrix[1:]       # Rest are documents
        
        similarities = cosine_similarity(question_vector, doc_vectors)[0]
        
        # Create (doc_id, similarity) pairs and sort by similarity
        doc_similarities = list(zip(valid_doc_ids, similarities))
        doc_similarities.sort(key=lambda x: x[1], reverse=True)
        
        # Return top documents
        top_docs = [doc_id for doc_id, _ in doc_similarities[:target_docs]]
        
        if self.verbose:
            print("TF-IDF similarity scores:")
            for doc_id, sim in doc_similarities[:min(10, len(doc_similarities))]:
                title = G.nodes[doc_id].get('title', doc_id)
                print(f"  {doc_id} ({title}): {sim:.3f}")
        
        return top_docs
    
    def _filter_with_embeddings(self, question: str, doc_ids: List[str], G: nx.Graph, 
                               target_docs: int) -> List[str]:
        """
        Filter documents using sentence embeddings and cosine similarity
        
        Args:
            question: The question text
            doc_ids: List of document IDs to filter
            G: NetworkX graph containing document nodes
            target_docs: Number of documents to keep
            
        Returns:
            List of filtered document IDs
        """
        if not EMBEDDINGS_AVAILABLE or self.embeddings_model is None:
            if self.verbose:
                print("Embeddings not available, falling back to TF-IDF")
            return self._filter_with_tfidf(question, doc_ids, G, target_docs)
        
        # Extract document texts
        doc_texts = []
        valid_doc_ids = []
        
        for doc_id in doc_ids:
            if doc_id in G.nodes and 'text' in G.nodes[doc_id]:
                doc_text = G.nodes[doc_id]['text']
                title = G.nodes[doc_id].get('title', '')
                # Combine title and text for better matching
                combined_text = f"{title} {doc_text}" if title else doc_text
                doc_texts.append(combined_text)
                valid_doc_ids.append(doc_id)
        
        if len(doc_texts) <= target_docs:
            return valid_doc_ids
        
        try:
            # Create embeddings for question and documents
            all_texts = [question] + doc_texts
            
            if self.verbose:
                print(f"Computing embeddings for {len(all_texts)} texts...")
            
            # Generate embeddings
            embeddings = self.embeddings_model.encode(all_texts, show_progress_bar=self.verbose)
            
            # Calculate cosine similarity between question and each document
            question_embedding = embeddings[0:1]  # First embedding is the question
            doc_embeddings = embeddings[1:]       # Rest are documents
            
            similarities = cosine_similarity(question_embedding, doc_embeddings)[0]
            
            # Create (doc_id, similarity) pairs and sort by similarity
            doc_similarities = list(zip(valid_doc_ids, similarities))
            doc_similarities.sort(key=lambda x: x[1], reverse=True)
            
            # Return top documents
            top_docs = [doc_id for doc_id, _ in doc_similarities[:target_docs]]
            
            if self.verbose:
                print("Embeddings similarity scores:")
                for doc_id, sim in doc_similarities[:min(10, len(doc_similarities))]:
                    title = G.nodes[doc_id].get('title', doc_id)
                    print(f"  {doc_id} ({title}): {sim:.3f}")
            
            return top_docs
            
        except Exception as e:
            if self.verbose:
                print(f"Embeddings filtering failed: {e}, falling back to TF-IDF")
            return self._filter_with_tfidf(question, doc_ids, G, target_docs)
    
    def _filter_with_llm(self, question: str, doc_ids: List[str], G: nx.Graph, 
                        target_docs: int) -> List[str]:
        """
        Filter documents using LLM-based relevance assessment
        
        Args:
            question: The question text
            doc_ids: List of document IDs to filter
            G: NetworkX graph containing document nodes
            target_docs: Number of documents to keep
            
        Returns:
            List of filtered document IDs
        """
        # Prepare document summaries for the LLM
        doc_summaries = []
        valid_doc_ids = []
        
        for doc_id in doc_ids:
            if doc_id in G.nodes and 'text' in G.nodes[doc_id]:
                title = G.nodes[doc_id].get('title', '')
                text = G.nodes[doc_id]['text']
                # Truncate text if too long
                if len(text) > 500:
                    text = text[:500] + "..."
                
                doc_summaries.append({
                    "id": doc_id,
                    "title": title,
                    "text": text
                })
                valid_doc_ids.append(doc_id)
        
        if len(doc_summaries) <= target_docs:
            return valid_doc_ids
        
        # Create prompt for LLM
        prompt = f"""
        I need to select the {target_docs} most relevant documents to answer this question:
        
        Question: {question}
        
        Available documents:
        {json.dumps(doc_summaries, indent=2)}
        
        Please analyze each document's relevance to the question and select the {target_docs} most relevant ones.
        Consider:
        1. Direct relevance to the question topic
        2. Presence of key entities mentioned in the question
        3. Information that could help answer the question
        4. Complementary information that works together
        
        Return only a JSON array of the selected document IDs, ordered by relevance (most relevant first).
        Example: ["doc_1", "doc_5", "doc_3"]
        """
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant that selects the most relevant documents for answering questions."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            response = self.deepseek.generate_response(messages, temperature=0.1)
            
            # Extract JSON from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                selected_docs = json.loads(json_str)
                
                # Validate that selected docs are in the original list
                valid_selected = [doc for doc in selected_docs if doc in valid_doc_ids]
                
                # If we don't have enough, fill with remaining docs
                if len(valid_selected) < target_docs:
                    remaining = [doc for doc in valid_doc_ids if doc not in valid_selected]
                    valid_selected.extend(remaining[:target_docs - len(valid_selected)])
                
                return valid_selected[:target_docs]
            else:
                if self.verbose:
                    print("Failed to parse LLM response, falling back to TF-IDF")
                return self._filter_with_tfidf(question, doc_ids, G, target_docs)
                
        except Exception as e:
            if self.verbose:
                print(f"LLM filtering failed: {e}, falling back to TF-IDF")
            return self._filter_with_tfidf(question, doc_ids, G, target_docs)
    
    def _filter_hybrid(self, question: str, doc_ids: List[str], G: nx.Graph, 
                      target_docs: int) -> List[str]:
        """
        Filter documents using a hybrid approach: TF-IDF + LLM validation
        
        Args:
            question: The question text
            doc_ids: List of document IDs to filter
            G: NetworkX graph containing document nodes
            target_docs: Number of documents to keep
            
        Returns:
            List of filtered document IDs
        """
        # First, use TF-IDF to get top candidates (more than target)
        tfidf_candidates = min(target_docs * 2, len(doc_ids))
        tfidf_filtered = self._filter_with_tfidf(question, doc_ids, G, tfidf_candidates)
        
        # If TF-IDF already gives us the target number or fewer, return it
        if len(tfidf_filtered) <= target_docs:
            return tfidf_filtered
        
        # Use LLM to make final selection from TF-IDF candidates
        if self.verbose:
            print(f"Hybrid filtering: TF-IDF selected {len(tfidf_filtered)} candidates, using LLM for final selection")
        
        return self._filter_with_llm(question, tfidf_filtered, G, target_docs)
    
    def analyze_filtering_impact(self, original_docs: List[str], filtered_docs: List[str], 
                               G: nx.Graph) -> Dict:
        """
        Analyze the impact of semantic filtering on document selection
        
        Args:
            original_docs: Original list of documents before filtering
            filtered_docs: Filtered list of documents
            G: NetworkX graph containing document nodes
            
        Returns:
            Dictionary with analysis results
        """
        removed_docs = [doc for doc in original_docs if doc not in filtered_docs]
        
        # Count supporting vs non-supporting documents
        original_supporting = sum(1 for doc in original_docs 
                                if G.nodes[doc].get('is_supporting', False))
        filtered_supporting = sum(1 for doc in filtered_docs 
                                if G.nodes[doc].get('is_supporting', False))
        removed_supporting = sum(1 for doc in removed_docs 
                               if G.nodes[doc].get('is_supporting', False))
        
        analysis = {
            "original_count": len(original_docs),
            "filtered_count": len(filtered_docs),
            "removed_count": len(removed_docs),
            "original_supporting": original_supporting,
            "filtered_supporting": filtered_supporting,
            "removed_supporting": removed_supporting,
            "supporting_retention_rate": filtered_supporting / original_supporting if original_supporting > 0 else 0,
            "precision_improvement": filtered_supporting / len(filtered_docs) if len(filtered_docs) > 0 else 0,
            "original_precision": original_supporting / len(original_docs) if len(original_docs) > 0 else 0
        }
        
        return analysis 