from dataclasses import dataclass
from typing import Callable


@dataclass
class FunctionDescriptor:
    name: str
    func: Callable
    json_repr: dict
    requires_context: bool
