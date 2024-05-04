from typing import List
from agent_dingo.rag.base import BaseReader as _BaseReader, Document
try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    raise ImportError(
        "requests or BeautifulSoup4 are not installed. Please install it using `pip install agent-dingo[rag_default]`"
    )


class WebpageReader(_BaseReader):
    def read(self, url: str) -> List[Document]:
        docs = []
        response = requests.get(url)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            text = soup.get_text(" ")
            docs.append(Document(text, {"source": url}))
        else:
            raise ValueError(f"Error fetching {url}: {response.status_code}")
        return docs
