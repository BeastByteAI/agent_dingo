from agent_dingo.rag.base import (
    BaseVectorStore as _BaseVectorStore,
    Chunk,
    RetrievedChunk,
)
from agent_dingo.utils import sha256_to_uuid

try:
    from qdrant_client import QdrantClient, AsyncQdrantClient
    from qdrant_client.http import models
except ImportError:
    raise ImportError(
        "Qdrant is not installed. Please install it using `pip install agenet-dingo[qdrant]`"
    )
from typing import Optional, List
from warnings import warn


class Qdrant(_BaseVectorStore):
    def __init__(
        self,
        collection_name: str,
        embedding_size: int,
        path: Optional[str] = None,
        host: Optional[str] = None,
        port: Optional[int] = None,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        recreate_collection: bool = False,
        upsert_batch_size: int = 32,
        try_init: bool = True,
    ):
        """
        Qdrant vector store.

        Parameters
        ----------
        collection_name : str
            collection name
        embedding_size : int
            size of the vector embeddings
        path : Optional[str], optional
            path to the local database, does not support concurrent clients, by default None
        host : Optional[str], optional
            host, by default None
        port : Optional[int], optional
            port, by default None
        url : Optional[str], optional
            base url of qdrant provider, by default None
        api_key : Optional[str], optional
            api key of qdrant provider, by default None
        recreate_collection : bool, optional
            flag to control whether the collection should be recreated on init, by default False
        upsert_batch_size : int, optional
            batch size for upserting the documents, by default 32
        try_init : bool, optional
            flag to control whether the collocetion should be created on object initialization, by default True

        Raises
        ------
        an
            _description_
        """
        clint_params = {
            "host": host,
            "path": path,
            "port": port,
            "url": url,
            "api_key": api_key,
        }
        client_params = {k: v for k, v in clint_params.items() if v is not None}
        self.client_params = client_params
        self._client = None
        self._async_client = None
        self.collection_name = collection_name
        self.embedding_size = embedding_size
        self.recreate_collection = recreate_collection
        self.upsert_batch_size = upsert_batch_size
        if path and (try_init or recreate_collection):
            warn(
                "Using local Qdrant storage will only work in a synchronous environment. Trying to call async methods will raise an error."
            )
        if try_init or recreate_collection:
            self._init_collection()

    def make_sync_client(self):
        self._client = (
            QdrantClient(**self.client_params)
            if self.client_params
            else QdrantClient(":memory:")
        )

    def make_async_client(self):
        self._async_client = (
            AsyncQdrantClient(**self.client_params)
            if self.client_params
            else AsyncQdrantClient(":memory:")
        )

    @property
    def client(self):
        if self._client is None:
            self.make_sync_client()
        return self._client

    @property
    def async_client(self):
        if self._async_client is None:
            self.make_async_client()
        return self._async_client

    def _init_collection(self):
        create_fn = (
            self.client.create_collection
            if not self.recreate_collection
            else self.client.recreate_collection
        )
        try:
            create_fn(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedding_size, distance=models.Distance.COSINE
                ),
            )
        except ValueError as e:
            if self.recreate_collection:
                raise e
            pass  # collection already exists

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
        return self._process_search_reults(search_result)

    async def async_retrieve(self, k: int, query: List[float]):
        search_result = await self.async_client.search(
            collection_name=self.collection_name,
            query_vector=query,
            limit=k,
        )
        return self._process_search_reults(search_result)

    def _process_search_reults(self, search_result: List) -> List[RetrievedChunk]:
        retrieved_chunks = []
        for r in search_result:
            content = r.payload["content"]
            metadata = r.payload["document_metadata"]
            score = r.score
            retrieved_chunks.append(RetrievedChunk(content, metadata, score))
        return retrieved_chunks
