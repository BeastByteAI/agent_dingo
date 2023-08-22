from agent_dingo.agent import AgentDingo
from agent_dingo.usage import UsageMeter
from agent_dingo.context import ChatContext
from typing import List, Optional, Any, Callable
from time import time
from uuid import uuid4


class DingoWrapper:
    """A wrapper for the Dingo agent that provides an OpenAI compatible API."""

    def __init__(
        self,
        agent: AgentDingo,
        before_function_call: Optional[Callable] = None,
        max_function_calls: int = 10,
    ):
        self.agent = agent
        self.before_function_call = before_function_call
        self.max_function_calls = max_function_calls

    def chat_completion(
        self,
        model: str,
        messages: List[dict],
        temperature: float = 0.7,
        chat_context: Optional[ChatContext] = None,
        **kwargs: Any,
    ) -> dict:
        """Chat with the provided agent using a similar API to OpenAI chat completion.

        Parameters
        ----------
        model : str
            The desired model to use (only used to reflect the same model in response, doesn't actually change behavior).
        messages : List[dict]
            A list of messages to send to the LLM.
        temperature : float, optional
            The temperature to use, by default 0.7.
        chat_context : Optional[ChatContext], optional
            The (dingo) chat context, by default None.

        Returns
        -------
        dict
            A formatted response similar to OpenAI's API.
        """
        meter = UsageMeter()
        response_content, _ = self.agent.chat(
            messages,
            model=model,
            temperature=temperature,
            before_function_call=self.before_function_call,
            usage_meter=meter,
            chat_context=chat_context,
            max_function_calls=self.max_function_calls,
            **kwargs,
        )

        finish_reason = meter.last_finish_reason or "stop"
        formatted_response = {
            "id": "chatcmpl-" + str(uuid4()),
            "object": "chat.completion",
            "created": int(time()),
            "model": "dingo-" + model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": response_content},
                    "finish_reason": finish_reason,
                }
            ],
            "usage": meter.get_usage(),
        }

        return formatted_response

    def create_app(self):
        """Creates a flask app that serves the agent."""
        from flask import Flask, request, jsonify

        app = Flask(__name__)

        @app.route("/chat/completions", methods=["POST"])
        def chat():
            data = request.json
            model = data["model"]
            messages = data["messages"]
            temperature = data.get("temperature", 0.7)
            response = self.chat_completion(model, messages, temperature)
            return jsonify(response)

        @app.route("/health", methods=["GET"])
        def health():
            return "The server is running."

        return app

    def serve(self, port: int = 8080, host: str = "0.0.0.0", threads: int = 4) -> None:
        """Serves the agent on a given port and host.

        Parameters
        ----------
        port : int, optional
            The port to serve on, by default 8080.
        host : str, optional
            The host to serve on, by default
        threads : int, optional
            The number of threads to use, by default 4.
        """
        from waitress import serve as waitress_serve

        app = self.create_app()
        msg = "The app is running on port " + str(port) + "."
        print(msg)
        waitress_serve(app, host=host, port=port, threads=threads)
