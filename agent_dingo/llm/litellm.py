from typing import Optional, Dict
from agent_dingo.core.blocks import BaseLLM
from agent_dingo.core.state import UsageMeter

try:
    from litellm import completion, acompletion
except ImportError:
    raise ImportError(
        "litellm is not installed. Please install it using `pip install agent-dingo[litellm]`"
    )


class LiteLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        completion_extra_kwargs: Optional[Dict] = None,
    ):

        self.temperature = temperature
        self.model = model
        self.completion_extra_kwargs = completion_extra_kwargs or {}

    def send_message(
        self,
        messages,
        functions=None,
        usage_meter: UsageMeter = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        response = completion(
            messages=messages,
            model=self.model,
            temperature=temperature or self.temperature,
            **self.completion_extra_kwargs,
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
        response = await acompletion(
            messages=messages,
            model=self.model,
            temperature=temperature or self.temperature,
            **self.completion_extra_kwargs,
        )
        self._log_usage(response, usage_meter)
        return response["choices"][0]["message"]

    def _log_usage(self, response, usage_meter: UsageMeter = None):
        if usage_meter:
            usage_meter.increment(
                prompt_tokens=response["usage"]["prompt_tokens"],
                completion_tokens=response["usage"]["completion_tokens"],
            )
