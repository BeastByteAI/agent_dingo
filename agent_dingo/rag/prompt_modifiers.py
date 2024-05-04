from agent_dingo.core.state import ChatPrompt, Context, Store
from agent_dingo.core.message import UserMessage, SystemMessage
from agent_dingo.core.blocks import BasePromptModifier as _BasePromptModifier
from agent_dingo.rag.base import BaseEmbedder, BaseVectorStore, RetrievedChunk
from typing import List, Optional
from warnings import warn

_DEFAULT_RAG_TEMPLATE = """
{original_message}

Relevant documents:
{documents}
"""


class RAGPromptModifier(_BasePromptModifier):
    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        n_chunks_to_retrieve: int = 5,
        retrieved_data_location: str = "system",
        rag_template: Optional[str] = None,
    ):
        if retrieved_data_location not in ["system", "user"]:
            raise ValueError(
                "retrieved_data_location must be one of 'system' or 'user'"
            )
        self.embedder = embedder
        self.vector_store = vector_store
        self.retrieved_data_location = retrieved_data_location
        self.rag_template = rag_template or _DEFAULT_RAG_TEMPLATE
        self.n_chunks_to_retrieve = n_chunks_to_retrieve

    def forward(self, state: ChatPrompt, context: Context, store: Store) -> ChatPrompt:
        if not isinstance(state, ChatPrompt):
            raise ValueError("state must be a ChatPrompt")
        query = state.messages[-1].content
        query_embedding = self.embedder.embed(query)[0]
        try:
            retrieved_data = self.vector_store.retrieve(
                self.n_chunks_to_retrieve,
                query_embedding,
            )
        except Exception as e:
            retrieved_data = []
            warn("No data was retrieved")
        return self._forward(state, retrieved_data)

    async def async_forward(
        self, state: ChatPrompt, context: Context, store: Store
    ) -> ChatPrompt:
        if not isinstance(state, ChatPrompt):
            raise ValueError("state must be a ChatPrompt")
        query = state.messages[-1].content
        query_embedding = (await self.embedder.async_embed(query))[0]
        try:
            retrieved_data = await self.vector_store.async_retrieve(
                self.n_chunks_to_retrieve,
                query_embedding,
            )
        except Exception as e:
            retrieved_data = []
            warn("No data was retrieved")
        return self._forward(state, retrieved_data)

    def _forward(
        self, state: ChatPrompt, retrieved_data: List[RetrievedChunk]
    ) -> ChatPrompt:
        if len(retrieved_data) < 1:
            return state
        modified = False
        messages = []
        target_message_type = (
            SystemMessage if self.retrieved_data_location == "system" else UserMessage
        )
        for message in state.messages:
            if isinstance(message, target_message_type) and not modified:
                modified_message = target_message_type(
                    self.rag_template.format(
                        original_message=message.content,
                        documents="\n".join([str(i.__dict__) for i in retrieved_data]),
                    )
                )
                modified = True
            else:
                modified_message = message.__class__(message.content)
            messages.append(modified_message)
        if not modified:
            raise ValueError(
                f"Could not find a {target_message_type.__name__} message to modify"
            )
        return ChatPrompt(messages)

    def get_required_context_keys(self) -> List[str]:
        return []
