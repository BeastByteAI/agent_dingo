from abc import ABC, abstractmethod
from agent_dingo.core.state import State, ChatPrompt, KVData


class BaseOutputParser(ABC):
    """Base class for output parsers."""

    def parse(self, output: State) -> str:
        """Delegates the parsing to the appropriate method based on the type of output.

        Parameters
        ----------
        output : State
            The state object to parse.

        Returns
        -------
        str
            parsed output
        """
        if not isinstance(output, State.__args__):
            raise TypeError(f"Expected KVData, got {type(output)}")
        elif isinstance(output, KVData):
            return self._parse_kvdata(output)
        else:
            return self._parse_chat(output)

    @abstractmethod
    def _parse_chat(self, output: ChatPrompt) -> str:
        pass

    @abstractmethod
    def _parse_kvdata(self, output: KVData) -> str:
        pass


class DefaultOutputParser(BaseOutputParser):
    """Default output parser. Expected output is a KVData with the key `_out_0`."""

    def _parse_chat(self, output: ChatPrompt) -> str:
        raise RuntimeError(
            "Cannot parse chat output. The output must be an instance of KVData."
        )

    def _parse_kvdata(self, output: KVData) -> str:
        if "_out_0" in output.keys():
            return output["_out_0"]
        else:
            raise KeyError("Could not find output in KVData.")
