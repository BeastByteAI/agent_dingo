try:
    from vertexai import init as _vertex_init
    from vertexai.generative_models import (
        Content,
        FunctionDeclaration,
        GenerativeModel,
        Part,
        Tool,
    )
except ImportError:
    raise ImportError(
        "VertexAI is not installed. Please install it using `pip install agent-dingo[vertexai]`"
    )
from typing import Optional, List
from agent_dingo.core.blocks import BaseLLM
from agent_dingo.core.state import UsageMeter
import json

_ROLES_MAP = {
    "user": "USER",
    "system": "USER",
    "assistant": "MODEL",
}


class Gemini(BaseLLM):
    def __init__(
        self, model: str, project: str, location: str, temperature: float = 0.7
    ):
        _vertex_init(project=project, location=location)
        self._model = GenerativeModel(model)
        self.supports_function_calls = True
        self.temperature = temperature

    def _get_tools(self, functions: Optional[List]) -> List[Tool]:
        if functions is None:
            return []
        declarations: List[FunctionDeclaration] = []
        for f in functions:
            declaration = FunctionDeclaration(
                name=f["name"],
                description=f["description"],
                parameters=f["parameters"],
            )
            declarations.append(declaration)
        tool = Tool(function_declarations=declarations)
        return [tool]

    def send_message(
        self,
        messages,
        functions=None,
        usage_meter: UsageMeter = None,
        temperature: Optional[float] = None,
        **kwargs,
    ):
        converted = self._openai_to_gemini(messages)
        out = self._model.generate_content(
            contents=converted,
            tools=self._get_tools(functions),
            generation_config={"temperature": temperature or self.temperature},
        )
        self._postprocess_response(out, usage_meter)

    async def async_send_message(
        self,
        messages,
        functions=None,
        usage_meter: UsageMeter = None,
        temperature=None,
        **kwargs,
    ):
        converted = self._openai_to_gemini(messages)
        response = await self._model.generate_content_async(
            contents=converted,
            tools=self._get_tools(functions),
            generation_config={"temperature": temperature or self.temperature},
        )
        return self._postprocess_response(response, usage_meter)

    def _postprocess_response(self, response, usage_meter: UsageMeter = None):
        n_prompt_tokens = response._raw_response.usage_metadata.prompt_token_count
        n_completion_tokens = (
            response._raw_response.usage_metadata.candidates_token_count
        )
        if usage_meter:
            usage_meter.increment(
                prompt_tokens=n_prompt_tokens,
                completion_tokens=n_completion_tokens,
            )
        return self._gemini_to_openai(response)

    def _openai_to_gemini(self, messages):
        """Converts OpenAI messages to VertexAI messages."""

        converted = []

        for message in messages:
            print("message -> ", message)
            if "_cache" in message.keys():
                converted.append(message["_cache"])
            elif message["role"] in _ROLES_MAP.keys():
                content = Content(
                    role=_ROLES_MAP[message["role"]],
                    parts=[
                        Part.from_text(message["content"]),
                    ],
                )
                converted.append(content)
            elif message["role"] == "tool":
                content = Content(
                    role="function",
                    parts=[
                        Part.from_function_response(
                            name=message["tool_call_id"],
                            response={
                                "content": message["content"],
                            },
                        )
                    ],
                )
                converted.append(content)
            else:
                raise ValueError(f"Invalid message {message}")
        return converted

    def _gemini_to_openai(self, response):
        """Converts the Gemini response to OpenAI response."""
        print(response)
        try:
            content = response.candidates[0].content.parts[0].text
        except AttributeError:
            content = None
        function_call = response.candidates[0].function_calls

        transformed_calls = []
        if len(function_call) > 0:
            for call in function_call:
                id_ = call.name
                type_ = "function"
                name = call.name
                args = {arg: call.args[arg] for arg in call.args}
                transformed_call = {
                    "id": id_,
                    "type": type_,
                    "function": {
                        "name": name,
                        "arguments": json.dumps(args),
                    },
                }
                transformed_calls.append(transformed_call)

        return {
            "content": content,
            "role": "assistant",
            "tool_calls": transformed_calls,
            "_cache": response.candidates[0].content,
        }
