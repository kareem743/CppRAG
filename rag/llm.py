import logging
import subprocess
from abc import ABC, abstractmethod


class LocalLLM(ABC):
    @abstractmethod
    def generate(self, prompt: str) -> str:
        raise NotImplementedError


class OllamaLLM(LocalLLM):
    def __init__(self, model: str, timeout_seconds: int = 60):
        self._model = model
        self._timeout_seconds = timeout_seconds

    def generate(self, prompt: str) -> str:
        logging.getLogger(__name__).info("Calling Ollama model '%s'", self._model)
        completed = subprocess.run(
            ["ollama", "run", self._model],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=self._timeout_seconds,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(f"Ollama failed: {stderr.strip()}")
        return completed.stdout.decode("utf-8", errors="ignore").strip()
