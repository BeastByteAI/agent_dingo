from typing import Optional, List
import openai

from tenacity import retry, stop_after_attempt, wait_fixed


@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
def send_message(
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

    Returns
    -------
    dict
        The response from the LLM.
    """
    f = {}
    if functions is not None:
        f["functions"] = functions
        f["function_call"] = "auto"
    response = openai.ChatCompletion.create(
        model=model, messages=messages, temperature=temperature, **f
    )
    return response["choices"][0]["message"]
