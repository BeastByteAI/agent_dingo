from typing import List

try:
    from PyPDF2 import PdfReader
except ImportError:
    raise ImportError(
        "PyPDF2 is not installed. Please install it using `pip install agent-dingo[rag_default]`"
    )
from agent_dingo.rag.base import BaseReader as _BaseReader, Document


class PDFReader(_BaseReader):
    def read(self, file_path: str) -> List[Document]:
        docs = []
        reader = PdfReader(file_path)
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            docs.append(Document(text, {"source": file_path, "page": i}))
        return docs
