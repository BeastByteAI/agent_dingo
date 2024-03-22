from typing import Optional, List
from agent_dingo.core.blocks import BaseLLM
from agent_dingo.core.state import UsageMeter
import openai
from tenacity import retry, stop_after_attempt, wait_fixed


@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
def _send_message(
    client: openai.OpenAI,
    messages: dict,
    model: str = "gpt-3.5-turbo-0613",
    functions: Optional[List] = None,
    temperature: float = 1.0,
) -> dict:
    """Sends messages to the LLM and returns the response.

    Parameters
    ----------
    messages : dict
        Messages to send to the LLM.
    model : str, optional
        Model to use, by default "gpt-3.5-turbo-0613"
    functions : Optional[List], optional
        List of functions to use, by default None
    temperature : float, optional
        Temperature to use, by default 1.
    log_usage : Callable, optional
        Function to log usage, by default None

    Returns
    -------
    dict
        The response from the LLM.
    """
    f = {}
    if functions is not None:
        f["tools"] = [{"type": "function", "function": f} for f in functions]
        f["tool_choice"] = "auto"
    response = client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, **f
    )
    return response.choices[0].message, response


@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
async def _async_send_message(
    client: openai.AsyncOpenAI,
    messages: dict,
    model: str = "gpt-3.5-turbo-0613",
    functions: Optional[List] = None,
    temperature: float = 1.0,
) -> dict:
    f = {}
    if functions is not None:
        f["tools"] = [{"type": "function", "function": f} for f in functions]
        f["tool_choice"] = "auto"
    response = await client.chat.completions.create(
        model=model, messages=messages, temperature=temperature, **f
    )
    return response.choices[0].message, response


def to_dict(obj):
    if isinstance(obj, dict):
        return {k: to_dict(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_dict(item) for item in obj]
    elif hasattr(obj, "__dict__"):
        return {k: to_dict(v) for k, v in obj.__dict__.items() if not k.startswith("_")}
    else:
        return obj


class OpenAI(BaseLLM):
    def __init__(
        self,
        model: str,
        temperature: float = 0.7,
        base_url: Optional[str] = None,
        # TODO: Add per instance API key
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
        response = _send_message(
            client=self.client,
            messages=messages,
            model=self.model,
            functions=functions,
            temperature=temperature or self.temperature,
        )
        return self._postprocess_response(response, usage_meter)

    async def async_send_message(
        self,
        messages,
        functions=None,
        usage_meter: UsageMeter = None,
        temperature=None,
        **kwargs,
    ):
        response = await _async_send_message(
            client=self.async_client,
            messages=messages,
            model=self.model,
            functions=functions,
            temperature=temperature or self.temperature,
        )
        return self._postprocess_response(response, usage_meter)

    def _postprocess_response(self, response, usage_meter: UsageMeter = None):
        res, full_res = to_dict(response[0]), to_dict(response[1])
        if usage_meter:
            usage_meter.increment(
                prompt_tokens=full_res["usage"]["prompt_tokens"],
                completion_tokens=full_res["usage"]["completion_tokens"],
            )
        if "function_call" in res.keys():
            del res["function_call"]
        return res
