from sentence_transformers import SentenceTransformer as _SentenceTransformer
from agent_dingo.rag.base import BaseEmbedder
from typing import List
import hashlib


class SentenceTransformer(BaseEmbedder):
    def __init__(self, model_name: str = "paraphrase-MiniLM-L6-v2"):
        self.model = _SentenceTransformer(model_name)
        self.model_name = model_name

    def embed(self, text: str) -> List[float]:
        return self.model.encode([text])[0].tolist()

    def hash(self) -> str:
        return hashlib.sha256(
            ("SentenceTransformer::" + self.model_name).encode()
        ).hexdigest()
