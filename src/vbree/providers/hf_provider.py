from huggingface_hub import InferenceClient
from .base import BaseProvider
from dotenv import load_dotenv
import os 
import time


load_dotenv()

class HfProvider(BaseProvider):
    def __init__(self, model: str, provider: str = "auto"):
        self.client = InferenceClient(
            provider=provider,
            api_key=os.getenv("HF_TOKEN")
        )
        self.model = model

    def generate(self, prompt: str, **kwargs) -> str:
        max_retries = 5
        wait_seconds = 10

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=1000,
                    **kwargs
                )
                return response.choices[0].message.content

            except Exception as e:
                error_msg = str(e)
                
                # network errors — worth retrying
                if any(err in error_msg for err in [
                    "getaddrinfo failed",
                    "WinError 10054", 
                    "ConnectError",
                    "ReadError",
                    "RemoteDisconnected"
                ]):
                    print(f"[{self.model}] Network error attempt {attempt + 1}/{max_retries}: {e}")
                    print(f"Waiting {wait_seconds}s before retry...")
                    time.sleep(wait_seconds)
                    wait_seconds *= 2  # exponential backoff
                    continue
                
                # other errors — not worth retrying
                print(f"[{self.model}] generation failed: {e}")
                return ""

        print(f"[{self.model}] All {max_retries} attempts failed")
        return ""