try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer
except ImportError:
    raise ImportError(
        "SentenceTransformers is not installed. Please install it using `pip install agent-dingo[sentence-transformers]`"
    )
from agent_dingo.rag.base import BaseEmbedder
from typing import List, Union
import hashlib
import concurrent.futures
import asyncio
import os


class SentenceTransformer(BaseEmbedder):
    def __init__(
        self, model_name: str = "paraphrase-MiniLM-L6-v2", batch_size: int = 128
    ):
        self.model = _SentenceTransformer(model_name)
        self.model_name = model_name
        self._executor = None
        self.batch_size = batch_size
        os.environ["TOKENIZERS_PARALLELISM"] = os.environ.get(
            "TOKENIZERS_PARALLELISM", "false"
        )

    def _prepare_executor(self) -> None:
        if self._executor is None:
            self._executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)

    async def async_embed(self, text: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(text, str):
            text = [text]
        self._prepare_executor()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.embed, text)

    def embed(self, texts: Union[str, List[str]]) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = self.model.encode(texts)
        return [embedding.tolist() for embedding in embeddings]

    def hash(self) -> str:
        return hashlib.sha256(
            ("SentenceTransformer::" + self.model_name).encode()
        ).hexdigest()
