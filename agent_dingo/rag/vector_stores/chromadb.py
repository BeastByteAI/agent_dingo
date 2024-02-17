from agent_dingo.rag.base import (
    BaseVectorStore as _BaseVectorStore,
    Chunk,
    RetrievedChunk,
)
from agent_dingo.utils import sha256_to_uuid
from typing import Optional, List
import chromadb


class ChromaDB(_BaseVectorStore):
    def __init__(
        self,
        collection_name: str,
        path: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        recreate_collection: bool = False,
        upsert_batch_size: int = 32,
    ):
        if path is not None and (host is not None or port is not None):
            raise ValueError("Either path or host/port must be specified, not both")
        if path is not None:
            self.client = chromadb.PersistentClient(path=path)
        else:
            self.client = chromadb.HttpClient(host=host, port=port)

        if recreate_collection:
            try:
                self.client.delete_collection(collection_name)
            except ValueError:
                pass

        self.collection = self.client.get_or_create_collection(collection_name)

        self.upsert_batch_size = upsert_batch_size

    def upsert_chunks(self, chunks: List[Chunk]):
        for i in range(0, len(chunks), self.upsert_batch_size):
            batch = chunks[i : i + self.upsert_batch_size]
            batch_contents = [chunk.content for chunk in batch]
            batch_ids = [sha256_to_uuid(chunk.hash) for chunk in batch]
            batch_embeddings = [chunk.embedding for chunk in batch]
            batch_metadata = [chunk.payload["document_metadata"] for chunk in batch]
            self.collection.upsert(
                ids=batch_ids,
                documents=batch_contents,
                embeddings=batch_embeddings,
                metadatas=batch_metadata,
            )

    def retrieve(self, k: int, query: List[float]):
        search_result = self.collection.query(
            query_embeddings=query,
            n_results=k,
            include=["metadatas", "documents", "distances"],
        )
        retrieved_chunks = []
        for content, metadata, score in zip(
            search_result["documents"][0],
            search_result["metadatas"][0],
            search_result["distances"][0],
        ):
            retrieved_chunks.append(RetrievedChunk(content, metadata, score))
        return retrieved_chunks
