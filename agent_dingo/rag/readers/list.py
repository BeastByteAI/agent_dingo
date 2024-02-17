from agent_dingo.rag.base import BaseReader as _BaseReader, Document
from typing import List, Optional


class ListReader(_BaseReader):
    def read(self, inputs: List[str]) -> List[Document]:
        docs = []
        for i in inputs:
            if not isinstance(i, str):
                raise ValueError("ListReader only accepts lists of strings")
            docs.append(Document(i, {"source": "memory"}))
        return docs
