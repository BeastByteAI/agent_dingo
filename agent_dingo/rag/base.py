from abc import ABC, abstractmethod
from typing import List, Optional
from dataclasses import dataclass
import hashlib
import json


@dataclass
class Document:
    content: str
    metadata: dict

    @property
    def hash(self) -> str:
        metadata = json.dumps(self.metadata, sort_keys=True)
        return hashlib.sha256((self.content + metadata).encode()).hexdigest()


@dataclass
class Chunk:
    content: str
    parent: Document
    embedding: Optional[List[str]] = None

    @property
    def payload(self):
        return {"content": self.content, "document_metadata": self.parent.metadata}

    @property
    def hash(self) -> str:
        parent_hash = self.parent.hash
        embdding_hash = (
            hashlib.sha256(json.dumps(self.embedding).encode()).hexdigest()
            if self.embedding
            else ""
        )
        content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        return hashlib.sha256(
            (parent_hash + embdding_hash + content_hash).encode()
        ).hexdigest()


@dataclass
class RetrievedChunk:
    content: str
    document_metadata: dict
    score: float


class BaseReader(ABC):
    @abstractmethod
    def read(self, *args, **kwargs) -> List[Document]:
        pass


class BaseChunker(ABC):
    @abstractmethod
    def chunk(self, document: Document) -> List[Chunk]:
        pass


class BaseEmbedder(ABC):
    # TODO: batch mode ?
    def embed_chunks(self, chunks: List[Chunk]):
        for chunk in chunks:
            chunk.embedding = self.embed(chunk.content)

    @abstractmethod
    def embed(self, text: str) -> List[float]:
        pass


class BaseVectorStore(ABC):
    @abstractmethod
    def upsert_chunks(self, chunks: List[Chunk]):
        pass

    @abstractmethod
    def retrieve(self, k: int, embedding: List[float]) -> List[RetrievedChunk]:
        pass
