from agent_dingo.context import ChatContext
from typing import Callable, List
import inspect


def construct_json_repr(
    name: str, description: str, properties: dict, required: List[str]
) -> dict:
    """Constructs a JSON representation of a function.

    Parameters
    ----------
    name : str
        The name of the function.
    description : str
        The description of the function.
    properties : dict
        The properties of the function (arguments, their descriptions and types).
    required : List[str]
        The required arguments of the function.

    Returns
    -------
    dict
        The JSON representation of the function.
    """
    return {
        "name": name,
        "description": description,
        "parameters": {
            "type": "object",
            "properties": properties,
        },
        "required": required,
    }


def get_required_args(func: Callable) -> List[str]:
    """Returns a list of the required arguments of a function.

    Parameters
    ----------
    func : Callable
        The function.

    Returns
    -------
    List[str]
        A list of the required arguments of the function.
    """
    sig = inspect.signature(func)
    params = sig.parameters
    required_args = [
        name
        for name, param in params.items()
        if param.default == inspect.Parameter.empty
        and not (name == "chat_context" and param.annotation == ChatContext)
    ]
    return required_args
