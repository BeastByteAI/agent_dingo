from agent_dingo.context import ChatContext
from typing import Callable, List
import inspect

def construct_json_repr(name: str, description: str, properties: dict, required: List[str]) -> dict:
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
    sig = inspect.signature(func)
    params = sig.parameters
    required_args = [
        name
        for name, param in params.items()
        if param.default == inspect.Parameter.empty
        and not (name == "chat_context" and param.annotation == ChatContext)
    ]
    return required_args
