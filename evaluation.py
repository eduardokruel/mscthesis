import re
import string
import json
import os
from difflib import SequenceMatcher
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# Download required NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

# Ensure WordNet is loaded before multi-threading
from nltk.corpus import wordnet as wn
wn.ensure_loaded()

class AnswerEvaluator:
    def __init__(self, alias_file=None):
        """
        Initialize the answer evaluator
        
        Args:
            alias_file: Optional path to a JSON file containing entity aliases
        """
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load entity aliases if provided
        self.aliases = {}
        if alias_file and os.path.exists(alias_file):
            with open(alias_file, 'r') as f:
                self.aliases = json.load(f)
    
    def normalize_answer(self, text):
        """
        Normalize answer text by removing punctuation, converting to lowercase,
        removing stop words, and lemmatizing
        
        Args:
            text: Answer text to normalize
            
        Returns:
            Normalized answer text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Tokenize
        tokens = word_tokenize(text)
        
        # Remove stop words and lemmatize
        try:
            tokens = [self.lemmatizer.lemmatize(token) for token in tokens if token not in self.stop_words]
        except Exception as e:
            # Fallback if lemmatization fails
            print(f"Warning: Lemmatization failed: {e}")
            tokens = [token for token in tokens if token not in self.stop_words]
        
        return ' '.join(tokens)
    
    def get_aliases(self, answer):
        """
        Get all possible aliases for an answer
        
        Args:
            answer: The reference answer
            
        Returns:
            List of possible aliases including the original answer
        """
        # Start with the original answer
        all_aliases = [answer]
        
        # Add any predefined aliases
        normalized_answer = self.normalize_answer(answer)
        if normalized_answer in self.aliases:
            all_aliases.extend(self.aliases[normalized_answer])
        
        # Generate common variations
        # 1. With/without "the"
        if answer.startswith("the "):
            all_aliases.append(answer[4:])
        else:
            all_aliases.append("the " + answer)
        
        # 2. With/without common titles
        titles = ["Dr.", "Professor", "Mr.", "Mrs.", "Ms."]
        for title in titles:
            if answer.startswith(title + " "):
                all_aliases.append(answer[len(title)+1:])
        
        # 3. First name / last name variations
        name_parts = answer.split()
        if len(name_parts) == 2:  # Likely a full name
            all_aliases.append(name_parts[0])  # First name
            all_aliases.append(name_parts[1])  # Last name
        
        # 4. Acronym expansion/contraction
        words = answer.split()
        if len(words) > 1:
            # Create acronym from multi-word answer
            acronym = ''.join(word[0].upper() for word in words if word[0].isalpha())
            if len(acronym) > 1:
                all_aliases.append(acronym)
        elif answer.isupper() and len(answer) > 1:
            # This might be an acronym, but we don't have expansion information
            pass
        
        # Remove duplicates and return
        return list(set(all_aliases))
    
    def exact_match(self, prediction, reference):
        """
        Check if prediction exactly matches reference or any of its aliases
        
        Args:
            prediction: Predicted answer
            reference: Reference answer
            
        Returns:
            Boolean indicating exact match
        """
        # Normalize both prediction and reference
        norm_prediction = self.normalize_answer(prediction)
        norm_reference = self.normalize_answer(reference)
        
        # Check for exact match with normalized reference
        if norm_prediction == norm_reference:
            return True
        
        # Check against aliases
        reference_aliases = self.get_aliases(reference)
        for alias in reference_aliases:
            norm_alias = self.normalize_answer(alias)
            if norm_prediction == norm_alias:
                return True
        
        return False
    
    def extract_answer_from_text(self, text):
        """
        Extract the most likely answer from generated text
        
        Args:
            text: Generated text that may contain the answer
            
        Returns:
            Extracted answer
        """
        # Look for common answer patterns
        patterns = [
            r"(?:The answer is|Answer:|A:|Answer is)[:\s]+([^\.]+)",
            r"(?:I believe|I think)[:\s]+([^\.]+)",
            r"(?:Based on|According to)[^,]+,\s+([^\.]+)"
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # If no pattern matches, return the first sentence as the answer
        sentences = re.split(r'[.!?]', text)
        if sentences:
            return sentences[0].strip()
        
        return text.strip()
    
    def evaluate(self, prediction, reference):
        """
        Evaluate a prediction against a reference answer, considering all possible aliases
        
        Args:
            prediction: Predicted answer text
            reference: Reference answer
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Extract answer from prediction if it's a longer text
        if len(prediction.split()) > 10:
            extracted_prediction = self.extract_answer_from_text(prediction)
        else:
            extracted_prediction = prediction
        
        # Check for exact match
        is_exact_match = self.exact_match(extracted_prediction, reference)
        
        # Normalize prediction
        norm_pred = self.normalize_answer(extracted_prediction)
        
        # Get all possible reference aliases and normalize them
        reference_aliases = self.get_aliases(reference)
        norm_aliases = [self.normalize_answer(alias) for alias in reference_aliases]
        
        # Add the original reference to the aliases if not already included
        norm_ref = self.normalize_answer(reference)
        if norm_ref not in norm_aliases:
            norm_aliases.append(norm_ref)
            reference_aliases.append(reference)
        
        # Calculate F1 scores for all aliases and select the best one
        best_f1 = 0.0
        best_precision = 0.0
        best_recall = 0.0
        best_alias = norm_ref
        best_alias_original = reference
        
        for i, norm_alias in enumerate(norm_aliases):
            # Calculate F1 score for token overlap
            pred_tokens = set(norm_pred.split())
            ref_tokens = set(norm_alias.split())
            
            if not pred_tokens or not ref_tokens:
                f1_score = 0.0
                precision = 0.0
                recall = 0.0
            else:
                common_tokens = pred_tokens.intersection(ref_tokens)
                precision = len(common_tokens) / len(pred_tokens) if pred_tokens else 0
                recall = len(common_tokens) / len(ref_tokens) if ref_tokens else 0
                f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0
            
            # Update best scores if this alias has a higher F1
            if f1_score > best_f1:
                best_f1 = f1_score
                best_precision = precision
                best_recall = recall
                best_alias = norm_alias
                best_alias_original = reference_aliases[i]
        
        # Check for substring relationship with the best alias
        is_substring = norm_pred in best_alias or best_alias in norm_pred
        
        # Determine if it's a partial match according to the criteria
        is_partial_match = best_f1 > 0.8 or (best_f1 > 0.6 and is_substring)
        
        # Calculate overall similarity using SequenceMatcher with the best alias
        similarity = SequenceMatcher(None, norm_pred, best_alias).ratio()
        
        # Return evaluation metrics
        return {
            "exact_match": is_exact_match,
            "partial_match": is_partial_match,
            "f1_score": best_f1,
            "precision": best_precision,
            "recall": best_recall,
            "is_substring": is_substring,
            "similarity": similarity,
            "extracted_prediction": extracted_prediction,
            "normalized_prediction": norm_pred,
            "normalized_reference": norm_ref,
            "best_matching_alias": best_alias_original,
            "best_matching_alias_normalized": best_alias
        }
    
    def add_alias(self, entity, alias):
        """
        Add an alias for an entity
        
        Args:
            entity: The entity name
            alias: The alias to add
        """
        normalized_entity = self.normalize_answer(entity)
        if normalized_entity not in self.aliases:
            self.aliases[normalized_entity] = []
        
        if alias not in self.aliases[normalized_entity]:
            self.aliases[normalized_entity].append(alias)
    
    def save_aliases(self, filepath):
        """
        Save aliases to a JSON file
        
        Args:
            filepath: Path to save the aliases
        """
        with open(filepath, 'w') as f:
            json.dump(self.aliases, f, indent=2)
    
    def load_aliases(self, filepath):
        """
        Load aliases from a JSON file
        
        Args:
            filepath: Path to the aliases file
        """
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                self.aliases = json.load(f) 