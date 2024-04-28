from typing import List
from agent_dingo.rag.base import BaseReader as _BaseReader, Document
import requests
from bs4 import BeautifulSoup


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
