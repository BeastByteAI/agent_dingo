from typing import Callable, Union, Optional, Tuple, List
from agent_dingo.parser import parse
from agent_dingo.helpers import get_required_args, construct_json_repr
from agent_dingo.context import ChatContext
from agent_dingo.chat import send_message
import json


class _Registry:
    def __init__(self):
        self.__functions = {}

    def add(
        self, name: str, func: Callable, json_repr: dict, requires_context: bool
    ) -> None:
        self.__functions[name] = {
            "func": func,
            "json_repr": json_repr,
            "requires_context": requires_context,
        }

    def get_function(self, name: str) -> Tuple[Callable, bool]:
        return (
            self.__functions[name]["func"],
            self.__functions[name]["requires_context"],
        )

    def get_available_functions(self) -> List[dict]:
        return [self.__functions[name]["json_repr"] for name in self.__functions]


class AgentDingo:
    def __init__(self):
        self._registry = _Registry()

    def register_function(self, func: Callable) -> None:
        docstring = func.__doc__
        if docstring is None:
            # TODO generate docstring
            raise ValueError("Function has no docstring")
        body, requires_context = parse(docstring)
        required_args = get_required_args(func)
        json_repr = construct_json_repr(
            func.__name__, body["description"], body["properties"], required_args
        )
        self._registry.add(func.__name__, func, json_repr, requires_context)
        # print("Registered function: ", func.__name__)
        # print(json_repr)

    def function(self, func: Callable) -> Callable:
        self.register_function(func)
        return func

    def chat(
        self,
        messages: Union[str, dict],
        chat_context=Optional[ChatContext],
        model: str = "gpt-3.5-turbo-0613",
        temperature: float = 1.0,
        max_function_calls: int = 10,
        before_function_call: Callable = None,
    ):
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        n_calls = 0
        available_functions = self._registry.get_available_functions()
        if not chat_context:
            chat_context = ChatContext()
        print(available_functions)
        while True:
            available_functions_i = (
                available_functions if n_calls < max_function_calls else None
            )
            response = send_message(
                messages,
                model=model,
                functions=available_functions_i,
                temperature=temperature,
            )
            if response.get("function_call"):
                function_name = response["function_call"]["name"]
                function_args = json.loads(response["function_call"]["arguments"])
                f, requires_context = self._registry.get_function(function_name)
                if requires_context:
                    function_args["chat_context"] = chat_context
                if before_function_call:
                    f, function_args = before_function_call(
                        function_name, f, function_args
                    )
                result = f(**function_args)
                messages.append(response)
                messages.append(
                    {
                        "role": "function",
                        "name": function_name,
                        "content": result,
                    }
                )
                n_calls += 1
            else:
                messages.append(response)
                return response["content"], messages
