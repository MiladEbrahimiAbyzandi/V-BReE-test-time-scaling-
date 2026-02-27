from .base import BaseProvider
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

class OpenAiProvider(BaseProvider):
    """Provider Implementation for OpenAI API."""

    def __init__(self, model: str):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model

    def generate (self, prompt: str, **kwargs) -> str:
        """Generate response from OpenAI API based on the given prompt."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs
        )
        return response.choices[0].message.content
    