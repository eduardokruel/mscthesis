# Import required libraries
from dotenv import load_dotenv
import os
from openai import OpenAI
from time import sleep

# Load environment variables
load_dotenv()

class DeepSeekAPI:
    def __init__(self):
        self.api_key = os.getenv('DEEPSEEK_API_KEY')
        if not self.api_key:
            raise ValueError("DEEPSEEK_API_KEY not found in environment variables")
        self.client = OpenAI(
            api_key=self.api_key,
            base_url="https://api.deepseek.com"
        )

    def generate_response(self, 
                         messages: list, 
                         temperature: float = 0.7,
                         max_retries: int = 3,
                         retry_delay: int = 1) -> str:
        """
        Generate a response using the DeepSeek API with retry mechanism
        """
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                    temperature=temperature,
                    stream=False
                )
                return response.choices[0].message.content
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                sleep(retry_delay * (attempt + 1))
                continue

def test_api():
    """Test the DeepSeek API connection"""
    try:
        deepseek = DeepSeekAPI()
        messages = [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": "What is 2+2?"}
        ]
        response = deepseek.generate_response(messages)
        print("API Test Response:", response)
        return True
    except Exception as e:
        print("API Test Error:", str(e))
        return False 