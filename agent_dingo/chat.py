from typing import Optional, List
import openai

from tenacity import retry, stop_after_attempt, wait_fixed

@retry(stop=stop_after_attempt(3), wait=wait_fixed(3))
def send_message(messages: dict, model: str = "gpt-3.5-turbo-0613", functions: Optional[List] = None, temperature: float = 1.):
    f = {}
    if functions is not None:
        f["functions"] = functions
        f["function_call"] = "auto"
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        **f
    )
    return response["choices"][0]["message"]
