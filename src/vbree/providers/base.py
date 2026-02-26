from abc import ABC, abstractmethod

class BaseProvider(ABC):
    """Abstract base class for all provider implementations."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Generate response from the model based on the given prompt."""
        pass 