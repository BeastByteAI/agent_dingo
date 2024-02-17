from agent_dingo.core.state import State, Store, Context, ChatPrompt
from agent_dingo.core.blocks import Pipeline
from agent_dingo.core.message import UserMessage, SystemMessage, AssistantMessage
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from typing import List, Dict, Optional, Tuple, Union
from uuid import uuid4
import time


class Message(BaseModel):
    role: str
    content: str


class PipelineRunRequest(BaseModel):
    model: str
    messages: List[Message]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Model(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "dingo"


class Models(BaseModel):
    models: List[Model]
    object: str = "list"


class Choice(BaseModel):
    index: int
    message: Message
    logprobs: Optional[Dict] = None
    finish_reason: str = "stop"


class PipelineOutputResponse(BaseModel):
    id: str
    object: str
    created: int
    model: str
    usage: Usage
    choices: List[Choice]


_role_to_message_type = {
    "user": UserMessage,
    "system": SystemMessage,
    "assistant": AssistantMessage,
}


def _construct_response(
    output: str, usage: Usage, model: str
) -> PipelineOutputResponse:
    generated_uuid = str(uuid4())
    current_timestamp = int(time.time())
    return PipelineOutputResponse(
        id=generated_uuid,
        object="chat.completion",
        created=current_timestamp,
        usage=usage,
        model=model,
        choices=[
            Choice(
                index=0,
                message=Message(role="assistant", content=output),
                finish_reason="stop",
            )
        ],
    )


def _construct_pipeline_input(
    input_: List[Message],
) -> Tuple[ChatPrompt, Dict[str, str]]:
    messages = []
    context = {}
    for m in input_:
        if m.role.startswith("context_"):
            key = m.role[8:]
            if key in context:
                raise ValueError(f"Context key {key} already exists.")
            context[key] = m.content
        else:
            msg = _role_to_message_type[m.role](m.content)
            messages.append(msg)
    state = ChatPrompt(messages)
    return state, context


def make_app(pipeline: Union[Pipeline, Dict[str, Pipeline]], is_async: bool = False):
    app = FastAPI()
    created_at = int(time.time())
    if isinstance(pipeline, Pipeline):
        available_pipelines = {"dingo": pipeline}
    else:
        for k, v in pipeline.items():
            if not isinstance(v, Pipeline):
                raise ValueError(f"Pipeline {k} is not an instance of Pipeline.")
        available_pipelines = pipeline

    if is_async:

        @app.post("/chat/completions")
        async def run_pipeline(input: PipelineRunRequest) -> PipelineOutputResponse:
            state, context = _construct_pipeline_input(input.messages)
            selected_pipeline = available_pipelines[input.model]
            output, usage = await selected_pipeline.async_run(_state=state, **context)
            return _construct_response(output, Usage(**usage), model=input.model)

    else:

        @app.post("/chat/completions")
        def run_pipeline(input: PipelineRunRequest) -> PipelineOutputResponse:
            state, context = _construct_pipeline_input(input.messages)
            selected_pipeline = available_pipelines[input.model]
            output, usage = selected_pipeline.run(_state=state, **context)
            return _construct_response(output, Usage(**usage), model=input.model)

    @app.get("/models")
    async def get_models() -> Models:
        models = Models(
            models=[Model(id=k, created=created_at) for k in available_pipelines.keys()]
        )
        return models

    return app


def serve_pipeline(
    pipeline: Union[Pipeline, Dict[str, Pipeline]],
    is_async: bool = False,
    host: str = "0.0.0.0",
    port: int = 8000,
):
    app = make_app(pipeline, is_async)
    uvicorn.run(app, host=host, port=port)
