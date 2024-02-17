from abc import ABC, abstractmethod
from agent_dingo.core.state import State, ChatPrompt, KVData


class BaseOutputParser(ABC):
    def parse(self, output: State) -> str:
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
    def _parse_chat(self, output: ChatPrompt) -> str:
        raise RuntimeError(
            "Cannot parse chat output. The output must be an instance of KVData."
        )

    def _parse_kvdata(self, output: KVData) -> str:
        if "_out_0" in output.keys():
            return output["_out_0"]
        else:
            raise KeyError("Could not find output in KVData.")
