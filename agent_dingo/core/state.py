from typing import Union, List
from agent_dingo.core.message import Message
from threading import Lock
from asyncio import Lock as AsyncLock


class ChatPrompt:
    def __init__(self, messages: List[Message]):
        self.messages = messages

    @property
    def dict(self):
        return [m.dict for m in self.messages]

    def __repr__(self):
        return f"ChatPrompt({self.messages})"


class KVData:
    def __init__(self, **kwargs):
        self._dict = {}
        for k, v in kwargs.items():
            self._dict[k] = v

    def update(self, key, value):
        if not isinstance(key, str) or not isinstance(value, str):
            raise TypeError("Both key and value must be strings.")
        # make existing keys immutable
        if key in self._dict:
            raise KeyError(f"Key {key} already exists.")
        self._dict[key] = value

    def __getitem__(self, key):
        return self._dict[key]

    def __repr__(self):
        return "KVData({0})".format(str(self._dict))

    def __dict__(self):
        return self._dict

    def keys(self):
        return self._dict.keys()

    def values(self):
        return self._dict.values()

    @property
    def dict(self):
        return self._dict.copy()


State = Union[ChatPrompt, KVData]


class Context(KVData):
    def update(self, key, value):
        raise RuntimeError("Context is immutable.")


class UsageMeter:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.last_finish_reason = None
        self._lock = Lock()

    def increment(self, prompt_tokens: int, completion_tokens: int) -> None:
        with self._lock:
            self.prompt_tokens += prompt_tokens
            self.completion_tokens += completion_tokens

    def get_usage(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }


class Store:
    def __init__(self):
        self._data = {}
        self._prompts = {}
        self._misc = {}
        self.usage_meter = UsageMeter()
        self._lock = Lock()  # probably not really needed

    def _update(self, key: str, item):
        if not isinstance(key, str):
            raise TypeError("Key must be a string.")
        if isinstance(item, ChatPrompt):
            self._prompts[key] = item
        elif isinstance(item, KVData):
            self._data[key] = item
        else:
            self._misc[key] = item

    def update(self, key: str, item):
        with self._lock:
            self._update(key, item)

    def get_misc(self, key: str):
        with self._lock:
            return self._misc[key]

    def get_data(self, key: str) -> KVData:
        with self._lock:
            return self._data[key]

    def get_prompt(self, key: str) -> ChatPrompt:
        with self._lock:
            return self._prompts[key]
