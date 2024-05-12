from dataclasses import dataclass
from typing import Callable, Optional, List


@dataclass
class FunctionDescriptor:
    name: str
    func: Callable
    json_repr: dict
    requires_context: bool
    required_context_keys: Optional[List[str]] = None
