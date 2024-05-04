import openai
from agent_dingo.rag.base import BaseEmbedder
from typing import Optional, List


class OpenAIEmbedder(BaseEmbedder):
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        base_url: Optional[str] = None,
        dimensions: Optional[int] = None,
    ):
        self.model = model
        self.client = openai.OpenAI(base_url=base_url)
        self.async_client = openai.AsyncOpenAI(base_url=base_url)
        self.params = {
            "model": self.model,
        }
        if dimensions:
            self.params["dimensions"] = dimensions

    def embed(self, texts: str) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        embeddings = [
            i.embedding
            for i in (self.client.embeddings.create(**self.params, input=texts).data)
        ]
        return embeddings

    async def async_embed(self, texts: str) -> List[List[float]]:
        if isinstance(texts, str):
            texts = [texts]
        res = await self.async_client.embeddings.create(**self.params, input=texts)
        embeddings = [i.embedding for i in res.data]
        return embeddings
