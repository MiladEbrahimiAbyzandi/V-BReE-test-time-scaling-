from huggingface_hub import InferenceClient
from .base import BaseProvider
from dotenv import load_dotenv
import os 

load_dotenv()

class HfProvider(BaseProvider):
    def __init__(self, model: str, provider: str = "auto"):
        self.client = InferenceClient(
            provider=provider,        
            api_key=os.getenv("HF_TOKEN")  
        )
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1000,
            **kwargs
        )
        return response.choices[0].message.content