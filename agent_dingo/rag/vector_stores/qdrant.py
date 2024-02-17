from agent_dingo.rag.base import (
    BaseVectorStore as _BaseVectorStore,
    Chunk,
    RetrievedChunk,
)
from agent_dingo.utils import sha256_to_uuid
from qdrant_client import QdrantClient
from qdrant_client.http import models
from typing import Optional, List


class Qdrant(_BaseVectorStore):
    def __init__(
        self,
        collection_name: str,
        embedding_size: int,
        host: Optional[str] = None,
        path: Optional[str] = None,
        port: Optional[int] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        recreate_collection: bool = False,
        upsert_batch_size: int = 32,
    ):
        clint_params = {
            "host": host,
            "path": path,
            "port": port,
            "url": url,
            "api_key": api_key,
        }
        client_params = {k: v for k, v in clint_params.items() if v is not None}
        self.client = (
            QdrantClient(**client_params) if client_params else QdrantClient(":memory:")
        )
        self.collection_name = collection_name
        self.embedding_size = embedding_size
        self.recreate_collection = recreate_collection
        self.upsert_batch_size = upsert_batch_size
        self._init_collection()

    def _init_collection(self):
        create_fn = (
            self.client.create_collection
            if not self.recreate_collection
            else self.client.recreate_collection
        )
        create_fn(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=self.embedding_size, distance=models.Distance.COSINE
            ),
        )

    def upsert_chunks(self, chunks: List[Chunk]):
        for i in range(0, len(chunks), self.upsert_batch_size):
            batch = chunks[i : i + self.upsert_batch_size]
            points = []
            for chunk in batch:
                if chunk.embedding is None:
                    raise ValueError("Chunk must be embedded before upserting")
                point = models.PointStruct(
                    vector=chunk.embedding,
                    payload=chunk.payload,
                    id=sha256_to_uuid(chunk.hash),
                )
                points.append(point)
            self.client.upsert(points=points, collection_name=self.collection_name)

    def retrieve(self, k: int, query: List[float]):
        search_result = self.client.search(
            collection_name=self.collection_name,
            query_vector=query,
            limit=k,
        )
        retrieved_chunks = []
        for r in search_result:
            content = r.payload["content"]
            metadata = r.payload["document_metadata"]
            score = r.score
            retrieved_chunks.append(RetrievedChunk(content, metadata, score))
        return retrieved_chunks
