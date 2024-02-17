from typing import Optional, List, Callable
from agent_dingo.core.blocks import BaseLLM
from agent_dingo.core.state import ChatPrompt, UsageMeter

import openai
from tenacity import retry, stop_after_attempt, wait_fixed


class FakeLLM(BaseLLM):
    def __init__(
        self,
        model: str = "123",
        temperature: float = 0.7,
        base_url: Optional[str] = None,
    ):
        self.model = model
        self.temperature = temperature
        self.client = openai.OpenAI(base_url=base_url)
        self.async_client = openai.AsyncOpenAI(base_url=base_url)
        if base_url is None:
            self.supports_function_calls = True

    def send_message(
        self,
        messages,
        functions=None,
        usage_meter: UsageMeter = None,
        temperature=None,
        **kwargs,
    ):
        res, full_res = (
            "Fake response",
            {
                "id": "chatcmpl-123",
                "object": "chat.completion",
                "created": 1677652288,
                "model": "gpt-3.5-turbo-0613",
                "system_fingerprint": "fp_44709d6fcb",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": "Fake response",
                        },
                        "logprobs": None,
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 9,
                    "completion_tokens": 12,
                    "total_tokens": 21,
                },
            },
        )
        if usage_meter:
            usage_meter.increment(
                prompt_tokens=full_res["usage"]["prompt_tokens"],
                completion_tokens=full_res["usage"]["completion_tokens"],
            )
        if "function_call" in res.keys():
            del res["function_call"]
        return res

    async def async_send_message(
        self,
        messages,
        functions=None,
        usage_meter: UsageMeter = None,
        temperature=None,
        **kwargs,
    ):
        return self.send_message(
            messages, functions, usage_meter, temperature, **kwargs
        )

    def process_prompt(
        self, prompt: ChatPrompt, usage_meter: Optional[UsageMeter] = None, **kwargs
    ):
        return self.send_message(prompt.dict, None, usage_meter)["content"]

    async def async_process_prompt(
        self, prompt: ChatPrompt, usage_meter: Optional[UsageMeter] = None, **kwargs
    ):
        return (await self.async_send_message(prompt.dict, None, usage_meter))[
            "content"
        ]
