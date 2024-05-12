from typing import Callable, List, Optional, Tuple


class Registry:
    """A registry for functions that can be called by the agent."""

    def __init__(self):
        self.__functions = {}
        self._required_context_keys = []

    def add(
        self,
        name: str,
        func: Callable,
        json_repr: dict,
        requires_context: bool,
        required_context_keys: Optional[List[str]] = None,
    ) -> None:
        """Adds a function to the registry.

        Parameters
        ----------
        name : str
            The name of the function.
        func : Callable
            The function.
        json_repr : dict
            The JSON representation of the function to be provided to the LLM.
        requires_context : bool
            Indicates whether the function requires a ChatContext object as one of its arguments.
        """
        if requires_context and required_context_keys is None:
            raise ValueError(
                "If requires_context is True, required_context_keys must be provided"
            )
        self.__functions[name] = {
            "func": func,
            "json_repr": json_repr,
            "requires_context": requires_context,
            "required_context_keys": required_context_keys or [],
        }

    def get_function(self, name: str) -> Tuple[Callable, bool]:
        """Retrieves a function from the registry.

        Parameters
        ----------
        name : str
            The name of the function.

        Returns
        -------
        Tuple[Callable, bool]
            A tuple containing the function and a boolean indicating whether the function requires a ChatContext object as one of its arguments.
        """
        try:
            return (
                self.__functions[name]["func"],
                self.__functions[name]["requires_context"],
            )
        except KeyError:
            return (
                (
                    lambda *args, **kwargs: f"Error: function `{name}` is not available. Most likely, the name is incorrect."
                ),
                False,
            )

    def get_available_functions(self) -> List[dict]:
        """Returns a list of JSON representations of the functions in the registry.

        Returns
        -------
        List[dict]
            A list of JSON representations of the functions in the registry.
        """
        return [self.__functions[name]["json_repr"] for name in self.__functions]

    def get_required_context_keys(self) -> List[str]:
        """Returns a list of keys that are required in the ChatContext object.

        Returns
        -------
        List[str]
            A list of keys that are required in the ChatContext object.
        """
        keys = []
        for f in self.__functions.values():
            keys.extend(f["required_context_keys"])
        return keys
