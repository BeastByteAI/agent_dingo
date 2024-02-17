from agent_dingo.rag.base import BaseChunker, Document, Chunk
from typing import List
import re


class RecursiveChunker(BaseChunker):
    def __init__(
        self, separators=None, chunk_size=512, keep_separator=False, merge_separator=" "
    ):
        if separators is None:
            separators = ["\n\n", "\n", " ", ""]
        self.separators = separators
        self.chunk_size = chunk_size
        self.keep_separator = keep_separator
        self.merge_separator = merge_separator

    def chunk(self, documents: List[Document]) -> List[Chunk]:
        all_chunks = []
        for doc in documents:
            chunks = self._split_text_recursive(doc.content, self.separators)
            chunks = self._merge_small_chunks(chunks)
            all_chunks.extend(
                [Chunk(content=chunk, parent=doc) for chunk in chunks]
            )
        return all_chunks

    def _split_text_recursive(self, text, separators):
        if not separators:
            return [text]

        separator = separators[0]
        pattern = re.escape(separator)
        split_chunks = re.split(pattern, text)

        final_chunks = []
        for i, chunk in enumerate(split_chunks):
            appended_chunk = (
                separator + chunk if self.keep_separator and i > 0 else chunk
            )
            if len(appended_chunk) <= self.chunk_size:
                final_chunks.append(appended_chunk)
            else:
                final_chunks.extend(
                    self._split_text_recursive(appended_chunk, separators[1:])
                )

        return final_chunks

    def _merge_small_chunks(self, chunks):
        merged_chunks = []
        current_chunk = ""

        for chunk in chunks:
            new_chunk = current_chunk + (
                self.merge_separator + chunk if current_chunk else chunk
            )
            if len(new_chunk) <= self.chunk_size:
                current_chunk = new_chunk
            else:
                if current_chunk:
                    merged_chunks.append(current_chunk)
                current_chunk = chunk

        if current_chunk:
            merged_chunks.append(current_chunk)

        return merged_chunks
