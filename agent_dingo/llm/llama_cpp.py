from typing import Optional
from agent_dingo.core.blocks import BaseLLM
from agent_dingo.core.state import UsageMeter

try:
    from llama_cpp import Llama as _Llama
except ImportError:
    raise ImportError(
        "Llama.cpp is not installed. Please install it using `pip install agent-dingo[llama-cpp]`"
    )
import threading
from concurrent.futures import ThreadPoolExecutor
import asyncio


class LlamaCPP(BaseLLM):
    def __init__(self, model: str, temperature: float = 0.7, verbose: bool = False):

        self.model = _Llama(model, verbose=verbose)
        self.temperature = temperature
        self._lock = threading.Lock()
        self._executor: Optional[ThreadPoolExecutor] = None

    def send_message(
        self,
        messages,
        functions=None,
        usage_meter: UsageMeter = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        with self._lock:
            response = self.model.create_chat_completion(
                messages, temperature=temperature or self.temperature
            )
        self._log_usage(response, usage_meter)
        return response["choices"][0]["message"]

    async def async_send_message(
        self,
        messages,
        functions=None,
        usage_meter: UsageMeter = None,
        temperature=None,
        **kwargs,
    ):
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            self._get_executor(),
            self.model.create_chat_completion,
            messages,
            temperature=temperature or self.temperature,
        )
        self._log_usage(response, usage_meter)
        return response["choices"][0]["message"]

    def _get_executor(self):
        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=1)
        return self._executor

    def _log_usage(self, response, usage_meter: UsageMeter = None):
        if usage_meter:
            usage_meter.increment(
                prompt_tokens=response["usage"]["prompt_tokens"],
                completion_tokens=response["usage"]["completion_tokens"],
            )
