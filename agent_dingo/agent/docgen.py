from typing import Callable
import inspect
from agent_dingo.core.blocks import BaseLLM
import re

_SYSTEM_MSG = "You are a code generation tool. Your responses are limited to providing the docstrings of functions."

_PROMPT = """
You will be provided with a Python function. Your task is to generate a docstring in a Google style for that python function and return only the docsting delimited by triple backticks. Do not return the function itself.

Python function:
```{code}```

Docstring:
"""


def generate_docstring(func: Callable, model: BaseLLM) -> str:
    """Generates a docstring for a given function.

    Parameters
    ----------
    func : Callable
        The function to generate a docstring for.
    model : str
        The model to use for generating the docstring.

    Returns
    -------
    str
        The generated docstring.
    """
    code = inspect.getsource(func)
    messages = [
        {"role": "system", "content": _SYSTEM_MSG},
        {"role": "user", "content": _PROMPT.format(code=code)},
    ]
    response = model.send_message(messages, temperature=0.0)

    response = (
        response["content"]
        .replace("```python\n", "")
        .replace("```", "")
        .replace('"""', "")
        .replace("'''", "")
    )

    return extract_substr(response)


def extract_substr(input_string: str) -> str:
    """Extracts the desription and args from a docstring.

    Parameters
    ----------
    input_string : str
        The docstring to extract the description and args from.

    Returns
    -------
    str
        Reduced docstring containing only the description and the args.
    """

    # Find the 'Returns:' string and capture everything before it
    match = re.search(r"(.*?)Returns:", input_string, re.DOTALL)

    if match:
        # Extract the portion before 'Returns:' and remove leading/trailing whitespace
        before_returns = match.group(1).strip()

        # Remove everything after 'Returns:' including the next line
        result = re.sub(r"Returns:.*?(\n|$)", "", before_returns, flags=re.DOTALL)
    else:
        result = input_string  # 'Returns:' string not found

    return result
