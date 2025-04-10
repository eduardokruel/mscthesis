# Import required libraries
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, BitsAndBytesConfig
from threading import Thread
import time
import json
from typing import List, Dict, Any, Optional, Union

class LocalLLM:
    """
    Class to handle local LLM inference using the Transformers library
    """
    def __init__(
        self, 
        model_name_or_path: str = "TheBloke/Llama-2-7B-Chat-GPTQ",
        device: str = "auto",
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        load_in_8bit: bool = False,
        load_in_4bit: bool = True,
        verbose: bool = False
    ):
        """
        Initialize the local LLM
        
        Args:
            model_name_or_path: HuggingFace model name or path to local model
            device: Device to run the model on ('auto', 'cuda', 'cpu')
            max_new_tokens: Maximum number of tokens to generate
            temperature: Temperature for sampling
            top_p: Top-p sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repeating tokens
            load_in_8bit: Whether to load the model in 8-bit precision
            load_in_4bit: Whether to load the model in 4-bit precision
            verbose: Whether to print verbose output
        """
        self.model_name = model_name_or_path
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.repetition_penalty = repetition_penalty
        self.verbose = verbose
        
        # Determine device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        if self.verbose:
            print(f"Using device: {self.device}")
        
        # Flag to track if model has been loaded
        model_loaded = False
        
        # Special handling for Qwen models
        if "qwen" in model_name_or_path.lower():
            # Load tokenizer with trust_remote_code=True for Qwen models
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path, 
                trust_remote_code=True
            )
            
            # Load the model with specific settings for Qwen
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                torch_dtype="auto",
                device_map="auto"
            )
            
            # Skip the later model loading
            model_loaded = True
        else:
            # Load tokenizer for non-Qwen models
            self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
            
            # Configure quantization parameters for non-Qwen models
            quantization_config = None
            
            # Check if model name contains GPTQ - use different loading method
            if "gptq" in model_name_or_path.lower():
                # For GPTQ models, we don't need BitsAndBytesConfig
                load_kwargs = {}
            else:
                # For regular models, use BitsAndBytesConfig
                if load_in_8bit:
                    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
                elif load_in_4bit:
                    quantization_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16
                    )
                
                load_kwargs = {
                    "quantization_config": quantization_config
                } if quantization_config else {}
            
            # Load the model for non-Qwen models
            if not model_loaded:
                # Load the model
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    device_map=self.device,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    **load_kwargs
                )
        
        # Set padding side to left for better batch processing
        self.tokenizer.padding_side = "left"
        
        # Add special tokens if needed
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        print(f"Model loaded successfully on {self.device}")
    
    def generate(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        """
        Generate a response using the local model
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature for generation (overrides instance value if provided)
            
        Returns:
            Generated text response
        """
        # Special handling for Qwen models
        if "qwen" in self.model_name.lower():
            try:
                # Don't add another system message if one already exists
                # The system message is already properly set for entity extraction
                
                # Format the messages using the tokenizer's chat template
                text = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                if self.verbose:
                    print(f"Formatted prompt:\n{text}")
                
                # Tokenize the formatted text
                model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)
                
                # Set up generation parameters - more conservative settings
                gen_kwargs = {
                    "max_new_tokens": self.max_new_tokens,
                    "temperature": 0.1,  # Lower temperature for more deterministic output
                    "top_p": 0.9,
                    "do_sample": False,  # Turn off sampling for deterministic output
                    "pad_token_id": self.tokenizer.eos_token_id,  # Use eos_token_id explicitly
                    "eos_token_id": self.tokenizer.eos_token_id,  # Set eos_token_id explicitly
                }
                
                # Generate response
                start_time = time.time()
                
                with torch.no_grad():
                    generated_ids = self.model.generate(
                        **model_inputs,
                        **gen_kwargs
                    )
                
                # Extract only the newly generated tokens
                generated_ids = [
                    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
                ]
                
                # Decode the response with explicit handling
                response = self.tokenizer.batch_decode(
                    generated_ids, 
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=True
                )[0]
                
                # Clean up any remaining strange characters
                response = self._clean_response(response)
                
                if self.verbose:
                    print(f"Generation took {time.time() - start_time:.2f} seconds")
                    print(f"Raw response:\n{response}")
                
                # For entity extraction, ensure we return valid JSON
                if "extract entities" in text.lower() or "entity" in text.lower():
                    # Try to extract JSON from the response
                    json_response = self._extract_json(response)
                    if json_response:
                        return json_response
                    
                    # If no JSON found but response contains entities, try to format it
                    if "[" in response and "]" in response:
                        try:
                            # Try to clean and format the response
                            cleaned = response.strip()
                            # Remove any text before the first [
                            if "[" in cleaned:
                                cleaned = cleaned[cleaned.find("["):]
                            # Remove any text after the last ]
                            if "]" in cleaned:
                                cleaned = cleaned[:cleaned.rfind("]")+1]
                            # Validate JSON
                            json.loads(cleaned)
                            return cleaned
                        except:
                            pass
                    
                    # If all else fails, try to extract entities manually
                    try:
                        # Look for patterns like "Entity1", "Entity2"
                        import re
                        entities = re.findall(r'"([^"]+)"', response)
                        if entities:
                            return json.dumps(entities)
                        
                        # If no quoted entities, try to find words that look like entities
                        words = re.findall(r'\b([A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*)\b', response)
                        if words:
                            return json.dumps(words)
                    except:
                        pass
                    
                    # Last resort: return empty array
                    return "[]"
                
                return response
                
            except Exception as e:
                if self.verbose:
                    print(f"Error using Qwen-specific generation: {e}")
                    print("Falling back to standard generation method")
                
                # Return a fallback JSON response for entity extraction
                if any("extract entities" in msg.get('content', '').lower() for msg in messages):
                    return "[]"  # Return empty entity list as fallback
        
        # Standard generation for non-Qwen models or if Qwen-specific method fails
        return self._generate_standard(messages, temperature)
    
    def _generate_standard(self, messages: List[Dict[str, str]], temperature: Optional[float] = None) -> str:
        """Standard generation method for non-Qwen models"""
        # Format the messages into a prompt
        prompt = self.format_messages(messages)
        
        if self.verbose:
            print(f"Prompt:\n{prompt}")
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Set up generation parameters
        gen_kwargs = {
            "max_new_tokens": self.max_new_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": True if (temperature if temperature is not None else self.temperature) > 0 else False,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        # Generate response
        start_time = time.time()
        
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                **gen_kwargs
            )
        
        # Decode the output
        response = self.tokenizer.decode(output[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        if self.verbose:
            print(f"Generation took {time.time() - start_time:.2f} seconds")
            print(f"Response:\n{response}")
        
        return response
    
    def format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format messages for the model
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            
        Returns:
            Formatted prompt string
        """
        # Detect model type and format accordingly
        if "qwen" in self.model_name.lower():
            return self._format_qwen_messages(messages)
        elif "llama" in self.model_name.lower():
            return self._format_llama_messages(messages)
        elif "mistral" in self.model_name.lower():
            return self._format_mistral_messages(messages)
        else:
            # Default format for other models
            return self._format_generic_messages(messages)
    
    def _format_qwen_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Qwen models"""
        try:
            # Try to use the model's built-in chat template
            chat_template = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            return chat_template
        except Exception as e:
            if self.verbose:
                print(f"Error applying chat template: {e}")
                print("Falling back to manual formatting")
            
            # Fall back to manual formatting
            formatted = ""
            
            # Add system message if present
            system_content = ""
            for msg in messages:
                if msg['role'] == 'system':
                    system_content += msg['content'] + "\n"
            
            if system_content:
                formatted += f"<|im_start|>system\n{system_content.strip()}<|im_end|>\n"
            
            # Add user/assistant messages
            for msg in messages:
                if msg['role'] == 'user':
                    formatted += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                elif msg['role'] == 'assistant':
                    formatted += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
            
            # Add final assistant prompt to get the response
            formatted += "<|im_start|>assistant\n"
            
            return formatted
    
    def _format_llama_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Llama models"""
        formatted = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                formatted += f"<|system|>\n{content}"
            elif role == 'user':
                if formatted:
                    # If we already have content, close the previous instruction
                    if not formatted.endswith(""):
                        formatted += f"{content} "
                    else:
                        formatted += f" {content} "
                else:
                    formatted += f" {content} "
            elif role == 'assistant':
                formatted += f" {content} "
        
        return formatted
    
    def _format_mistral_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages for Mistral models"""
        formatted = ""
        for i, msg in enumerate(messages):
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                formatted += f" {content} "
            elif role == 'user':
                if i > 0 and messages[i-1]['role'] == 'assistant':
                    formatted += f" {content} "
                else:
                    formatted += f" {content} "
            elif role == 'assistant':
                formatted += f" {content} "
        
        return formatted
    
    def _format_generic_messages(self, messages: List[Dict[str, str]]) -> str:
        """Generic message formatting for other models"""
        formatted = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            
            if role == 'system':
                formatted += f"System: {content}\n\n"
            elif role == 'user':
                formatted += f"User: {content}\n\n"
            elif role == 'assistant':
                formatted += f"Assistant: {content}\n\n"
        
        formatted += "Assistant: "
        return formatted
    
    def generate_streaming(self, messages: List[Dict[str, str]], temperature: Optional[float] = None):
        """
        Generate a response with streaming output
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature for generation (overrides instance value if provided)
            
        Returns:
            Iterator yielding generated text chunks
        """
        # Format the messages into a prompt
        prompt = self.format_messages(messages)
        
        # Tokenize the prompt
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        # Set up the streamer
        streamer = TextIteratorStreamer(self.tokenizer, skip_special_tokens=True)
        
        # Set up generation parameters
        gen_kwargs = {
            "input_ids": inputs["input_ids"],
            "max_new_tokens": self.max_new_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repetition_penalty": self.repetition_penalty,
            "do_sample": True if (temperature if temperature is not None else self.temperature) > 0 else False,
            "pad_token_id": self.tokenizer.pad_token_id,
            "streamer": streamer,
        }
        
        # Start generation in a separate thread
        thread = Thread(target=self.model.generate, kwargs=gen_kwargs)
        thread.start()
        
        # Yield from the streamer
        for text in streamer:
            yield text
        
        # Wait for the thread to finish
        thread.join()
    
    def _clean_response(self, text: str) -> str:
        """Clean up response text to remove strange characters"""
        # Replace common problematic characters
        replacements = {
            '桤': '',
            'etzt': '',
            '枃': '',
            # Add more problematic characters as needed
        }
        
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        
        # Remove any non-ASCII characters if still problematic
        # text = ''.join(c for c in text if ord(c) < 128)
        
        return text.strip()
    
    def _extract_json(self, text: str) -> str:
        """Try to extract JSON from text"""
        try:
            # Look for JSON array pattern
            if '[' in text and ']' in text:
                start = text.find('[')
                end = text.rfind(']') + 1
                json_str = text[start:end]
                # Validate by parsing
                json.loads(json_str)
                return json_str
            return ""
        except:
            return ""

# Function to test the local LLM
def test_local_llm(model_name: str = "TheBloke/Llama-2-7B-Chat-GPTQ"):
    """Test the local LLM with a simple prompt"""
    try:
        llm = LocalLLM(model_name_or_path=model_name, verbose=True)
        
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, are you working?"}
        ]
        
        print(f"Testing with model: {model_name}")
        response = llm.generate(messages)
        print(f"LLM Response: {response}")
        return True, llm
    except Exception as e:
        print(f"LLM Test Error: {e}")
        return False, None 