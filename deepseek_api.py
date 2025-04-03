# Import required libraries
from dotenv import load_dotenv
import os
import time
import threading
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from rate_limiter import get_rate_limiter

# Load environment variables
load_dotenv()

# Global semaphore to limit concurrent API requests
_api_semaphore = threading.BoundedSemaphore(100)  # Default to 100 concurrent requests

# Get the rate limiter
rate_limiter = get_rate_limiter()

def set_max_concurrent_requests(max_requests):
    """Set the maximum number of concurrent API requests"""
    global _api_semaphore
    _api_semaphore = threading.BoundedSemaphore(max_requests)

class DeepSeekAPI:
    def __init__(self, model_name="deepseek-chat"):
        """
        Initialize the API client
        
        Args:
            model_name: The model to use. Options:
                - "deepseek-chat": DeepSeek's chat model
                - "gpt-4o-mini": OpenAI's GPT-4o-mini model
                - Any other OpenAI model name
        """
        self.model_name = model_name
        
        # Check if using DeepSeek or OpenAI
        self.is_deepseek = "deepseek" in model_name.lower()
        
        # Get appropriate API key
        if self.is_deepseek:
            self.api_key = os.environ.get("DEEPSEEK_API_KEY")
            if not self.api_key:
                raise ValueError("DEEPSEEK_API_KEY environment variable not set")
            api_base = 'https://api.deepseek.com'
        else:
            self.api_key = os.environ.get("OPENAI_API_KEY")
            if not self.api_key:
                raise ValueError("OPENAI_API_KEY environment variable not set")
            api_base = None  # Use default OpenAI base URL
        
        # Initialize the LangChain model
        self.llm = BaseChatOpenAI(
            model=model_name,
            openai_api_key=self.api_key,
            openai_api_base=api_base,
            request_timeout=60  # Increase timeout to 60 seconds
        )
    
    def generate_response(self, messages, temperature=0.7, max_retries=3, retry_delay=2):
        """
        Generate a response using the selected model with rate limiting
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            temperature: Temperature for generation
            max_retries: Maximum number of retries on failure
            retry_delay: Initial delay between retries (doubles with each retry)
            
        Returns:
            Generated text response
        """
        retry_count = 0
        while retry_count < max_retries:
            # Check rate limits
            can_request, wait_time = rate_limiter.check_and_update(self.model_name)
            
            if not can_request:
                print(f"Rate limit reached for {self.model_name}. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time + 0.5)  # Add a small buffer
                continue
            
            try:
                # Convert messages to LangChain format
                langchain_messages = []
                for msg in messages:
                    if msg['role'] == 'system':
                        langchain_messages.append(SystemMessage(content=msg['content']))
                    elif msg['role'] == 'user':
                        langchain_messages.append(HumanMessage(content=msg['content']))
                
                # Set temperature
                self.llm.temperature = temperature
                
                # Generate response
                response = self.llm.invoke(langchain_messages)
                
                # Return the content
                return response.content
                    
            except Exception as e:
                error_str = str(e)
                print(f"Exception in API call: {error_str}")
                
                # Check if it's a rate limit error
                if "rate_limit" in error_str.lower() or "429" in error_str:
                    # Parse wait time if available
                    import re
                    wait_match = re.search(r'Please try again in (\d+\.\d+)s', error_str)
                    if wait_match:
                        wait_time = float(wait_match.group(1))
                    else:
                        wait_time = retry_delay * (2 ** retry_count)  # Exponential backoff
                    
                    print(f"Rate limit exceeded. Waiting {wait_time} seconds...")
                    time.sleep(wait_time + 0.5)  # Add a small buffer
                else:
                    # For other errors, use normal retry logic
                    retry_count += 1
                    if retry_count < max_retries:
                        print(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        return f"Error: {error_str}"
        
        return "Error: Maximum retries exceeded"

def test_api(model_name="deepseek-chat"):
    """Test the API connection with the specified model"""
    try:
        api = DeepSeekAPI(model_name=model_name)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello, are you working?"}
        ]
        
        print(f"Testing with model: {model_name}")
        response = api.generate_response(messages)
        print(f"API Response: {response}")
        return True
    except Exception as e:
        print(f"API Test Error: {e}")
        return False 