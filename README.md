<h1 align="center">
  <br>
 <img src="https://github.com/OKUA1/agent_dingo/blob/main/logo.png?raw=true" alt="AgentDingo" width="250" height = "250">
  <br>
  Agent Dingo
  <br>
</h1>

<h4 align="center">A microframework for building LLM-based pipelines and agents.</h4>

<p align="center">
  <a href="https://github.com/OKUA1/agent_dingo/releases">
    <img src="https://img.shields.io/github/v/release/OKUA1/agent_dingo.svg">
  </a>
  <a href="https://opensource.org/licenses/MIT">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg">
  </a>
  <a href="https://discord.gg/YDAbwuWK7V">
    <img src="https://dcbadge.vercel.app/api/server/YDAbwuWK7V?compact=true&style=flat">
  </a>
    <a href="https://medium.com/@iryna230520">
    <img src="https://img.shields.io/badge/Medium-%23000000.svg?&style=flat&logo=Medium&logoColor=white">
  </a>
</p>

_Dingo_ allows you to easily integrate any function into ChatGPT by adding a single line of code. With _Dingo_, you no longer have to worry about manually integrating the functions or dealing with intermediate function calls. The framework is crafted to automate these tasks, allowing you to focus on writing the core functionality.

## Quick Start ⚡️

> ⚠️ **This is a `v1` branch which is still in development**. The documentation might be incomplete, and some features might not be fully implemented. The current stable version is `v0.1.0`, which can be found in the [`main`](https://github.com/BeastByteAI/agent_dingo) branch.

**Step 1:** Install `agent-dingo`

```bash
pip install git+https://github.com/beastbyteai/agent_dingo.git@pipelines
```

**Step 2:** Configure your OpenAI API key

```bash
export OPENAI_API_KEY=<YOUR_KEY>
```

**Step 3:** Instantiate the `LLM` and `Agent`

```python
from agent_dingo.agent import Agent
from agent_dingo.llm.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")
agent = Agent(llm)
```

**Step 4:** Add `agent.function` decorator to the function you wish to integrate

```python
@agent.function
def get_current_weather(city: str):
    ...
```

**Step 5:** Run the agent

```python
from agent_dingo.core.message import UserMessage
from agent_dingo.core.state import ChatPrompt

# Pipeline with raw chat prompt
pipeline_raw = agent.as_pipeline()
prompt = ChatPrompt(messages=[UserMessage("What is the current weather in Linz?")])
out = pipeline_raw.run(prompt)
print(out)

# Pipeline with a prompt builder
from agent_dingo.core.blocks import PromptBuilder
prompt_template = [UserMessage("What is the current weather in {city}?")]
prompt_builder = PromptBuilder([UserMessage("What is the current weather in {city}?")])
pipeline_with_builder = prompt_builder >> agent
out = pipeline_with_builder.run(city="Linz")
print(out)
```

**Optional:** Run an OpenAI compatible server

```python
from agent_dingo.serve import serve_pipeline
serve_pipeline({"weather_model_1": pipeline_raw, "weather_model_2": pipeline_with_builder}, port = 8080)
```

The server can be accessed using the `openai` python package:

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8080")

# raw pipeline
messages = [
    {"role": "user", "content": "What is the current weather in Linz?"},
]

out = client.chat.completions.create(messages=messages, model="weather_model_1")
print(out)

# pipeline with prompt builder
messages = [
    {"role": "context_city", "content": "Linz"},
]

out = client.chat.completions.create(messages=messages, model="weather_model_2")
print(out)
```

## Support us 🤝

You can support the project in the following ways:

⭐ Star _Dingo_ on GitHub (click the star button in the top right corner)

💡 Provide your feedback or propose ideas in the issues section or Discord

📰 Post about _Dingo_ on LinkedIn or other platforms

🔗 Check out our other projects (cards below are clickable):

<a href="https://github.com/OKUA1/falcon"><img src="https://raw.githubusercontent.com/gist/OKUA1/6264a95a8abd225c74411a2b707b0242/raw/3cedb53538cb04656cd9d7d07e697e726896ce9f/falcon_light.svg"/></a> <br>
<a href="https://github.com/iryna-kondr/scikit-llm"><img src="https://gist.githubusercontent.com/OKUA1/6264a95a8abd225c74411a2b707b0242/raw/029694673765a3af36d541925a67214e677155e5/skllm_light.svg"/></a>

## Documentation 📚

For now, the documentation only covers the core concepts and does not provide a comprehensive overview of the internal workings of the framework. The full documentation will be released once the `v1` is stable.

### Blocks and Pipelines

The core concept of the framework is the `Block` and `Pipeline`. The `Block` is a single element of the pipeline that can be used to perform a specific task (e.g. build a prompt, call an LLM, etc.). Each block has a `forward` method that receives the current state, a global immutable context and a global mutable store. The block modifies and returns the state. The `Pipeline` is a sequence of blocks that are executed in order.

The simplest pipeline consists of a single `LLM` block, which is used to call the LLM model.

```python
from agent_dingo.llm.openai import OpenAI
from agent_dingo.core.message import UserMessage
from agent_dingo.core.state import ChatPrompt
llm = OpenAI(model="gpt-3.5-turbo") # this expects the OPENAI_API_KEY environment variable to be set

prompt = ChatPrompt(messages=[UserMessage("What is the population of Linz?")])
pipeline = llm.as_pipeline()
pipeline.run(prompt) # the first (optional) argument is the initial state of the pipeline
```

#### Building a pipeline with several blocks

The pipeline can be built by simply concatenating the blocks using the `>>` operator. For example, instead of providing the whole `ChatPrompt` as an initial state, we want to only provide the city and build the prompt inside the pipeline.

```python
from agent_dingo.core.blocks import PromptBuilder
from agent_dingo.llm.openai import OpenAI
from agent_dingo.core.message import UserMessage
from agent_dingo.core.state import ChatPrompt

llm = OpenAI(model="gpt-3.5-turbo")

template = [UserMessage("What is the population of {city}?")]
prompt_builder = PromptBuilder(template=template)

pipeline = prompt_builder >> llm

pipeline.run(city="Linz")
```

In this example, the command `pipeline.run(city="Linz")` is equivalent to `pipeline.run(state_ = None, city = Linz)`, where the additional keyword arguments (in this case `city`) are used to form a **global context** of the pipeline, which can be accessed by all blocks. By default, the `PromptBuilder` block will attempt to extract all the required arguments from the global context.

#### Parallel branches

It is also possible to run several blocks in parallel. For example, we might want to call different LLMs with the same prompt and then compare the results.

```python
from agent_dingo.core.blocks import PromptBuilder, Squeeze
from agent_dingo.llm.openai import OpenAI
from agent_dingo.core.message import UserMessage
from agent_dingo.core.state import ChatPrompt

gpt3 = OpenAI(model="gpt-3.5-turbo")
gpt4 = OpenAI(model="gpt-4")

squeeze_template_string = "gpt3: {gpt3_result}\ngpt4: {gpt4_result}"

template = [UserMessage("What is the population of {city}?")]
prompt_builder = PromptBuilder(template=template)

pipeline = prompt_builder >> (gpt3 & gpt4) >> Squeeze(squeeze_template_string)
pipeline.run(city="Linz")
```

The `&` operator initializes a `Parallel` block that runs the `gpt3` and `gpt4` blocks in parallel. The `Squeeze` block is used to combine the results of the parallel branches into a single string.

#### Custom and inline blocks

It is possible to define a custom block by inheriting from the `Block` class and implementing the `forward` method. However, it is often easier to use the `InlineBlock` decorator which allows to define the block as a simple function.

```python
from agent_dingo.core.blocks import PromptBuilder, Squeeze
from agent_dingo.core.message import UserMessage
from agent_dingo.core.state import ChatPrompt

@InlineBlock()
def fake_llm(state, context, store):
    return "This is a fake response."

pipeline = fake_llm.as_pipeline()
prompt = ChatPrompt(messages=[UserMessage("What is the population of Linz?")])
pipeline.run(prompt)
```

#### Async pipelines

The `Pipeline` can be run asynchronously using the `async` method, which in turn uses the `async_forward` method of the blocks. When defining an inline block, it is sufficient to use an `async` keyword in the function definition.

```python
from agent_dingo.core.blocks import PromptBuilder, Squeeze
from agent_dingo.core.message import UserMessage
from agent_dingo.core.state import ChatPrompt
import asyncio

@InlineBlock()
async def fake_llm(state, context, store):
    return "This is a fake response."

pipeline = fake_llm.as_pipeline()

asyncio.run(pipeline.async_run(ChatPrompt(messages=[UserMessage("What is the population of Linz?")])))
```

If async and sync blocks are mixed in the same pipeline, the runtime will still be able to run the pipeline (in both sync and async modes), but produce a warning as this scenario is not performance-optimized and not recommended for production use.

### Agents

`Agent` is a block which allows you to register the functions to use with the LLM. The intermediate function calls are handled automatically by the agent. The function can be registered with a single line of code using the `agent.function` decorator, which generates the function descriptor from the docstring.

```python
from agent_dingo.agent import Agent
from agent_dingo.llm.openai import OpenAI
import requests

llm = OpenAI(model="gpt-3.5-turbo")
agent = Agent(llm, max_function_calls=3)

@agent.function # equivalent to `agent.register_function(get_temperature)`
def get_temperature(city: str) -> str:
    """Retrieves the current temperature in a city.

    Parameters
    ----------
    city : str
        The city to get the temperature for.

    Returns
    -------
    str
        String representation of the json response from the weather api.
    """
    base_url = "https://api.openweathermap.org/data/2.5/weather"
    params = {
        "q": city,
        "appid": "<openweathermap_api_key>",
        "units": "metric"
    }
    response = requests.get(base_url, params=params)
    data = response.json()
    return str(data)

pipeline = agent.as_pipeline()
...

```

If the function does not have a docstring, it can still be registered. In that case, the docstring will be generated automatically by ChatGPT.
However, there are several drawbacks of this approach:

- The source code of your function is passed to the model;
- The generated docstring might be inaccurate (especially for complex functions);
- The generated docstrings are not persistant (i.e. they are not saved to disk) and will have to be regenerated every time the function is registered.

If you wish to disable the automatic docstring generation, you can set the `allow_codegen` parameter to `False` when instantiating the agent.

By default the `allow_codegen` parameter is set to `"env"` which means that the value is read from the `DINGO_ALLOW_CODEGEN` environment variable. If the variable is not set, it is assumed to be `True`.

#### LangChain Tools 🦜️🔗

It is possible to convert [Langchain Tools](https://python.langchain.com/docs/modules/agents/tools/) into dingo-compatible functions (refered to as function descriptors) in order to register them with Dingo. The converter can be used as follows:

1. Install langchain:

```bash
pip install agent_dingo[langchain]
```

2. Define the tool, we will use the Wikipedia tool as an example:

```python
from langchain.tools.wikipedia.tool import WikipediaQueryRun
from langchain.utilities.wikipedia import WikipediaAPIWrapper
tool = WikipediaQueryRun(api_wrapper = WikipediaAPIWrapper())
```

Please refer to the [LangChain documentation](https://python.langchain.com/docs/modules/agents/tools/) for more details on how to define the tools.

3. Convert the tool into a function descriptor and register:

```python
from agent_dingo.agent.langchain import convert_langchain_tool
descriptor = convert_langchain_tool(tool)
agent.register_descriptor(descriptor)
```

#### Multiple Agents / Sub-agents

Each agent can be used as a function descriptor, which allows to create a hierarchy of agents. This can be useful when you want to create several specialized agents that can be used from a manager agent when appropriate.

```python
from agent_dingo.core.message import UserMessage, SystemMessage, AssistantMessage
from agent_dingo.core.state import State, ChatPrompt, KVData, Context, Store
from agent_dingo.core.blocks import (
    PromptBuilder,
)
from agent_dingo.agent import Agent
from agent_dingo.llm.openai import OpenAI


llm = OpenAI(model="gpt-3.5-turbo")


messages = [
    SystemMessage("You are a helpful assistant. Provide concise anwers."),
    UserMessage("What is the weather in {city}?"),
]

qa_prompt = PromptBuilder(messages)

agent = Agent(
    llm,
    max_function_calls=2,
    name="manager",
    description="An agent with access to assisting agents.",
)

another_agent = Agent(
    llm,
    max_function_calls=2,
    name="another_agent",
    description="An agent with access to weather forecast data.",
)


@another_agent.function
def get_temperature(city: str) -> str:
    """Retrieves the current temperature in a city.

    Parameters
    ----------
    city : str
        The city to get the temperature for.

    Returns
    -------
    str
        str representation of the json response from the weather api.
    """
    return '{{"temperature"}}:"20"'  # fake response


agent.register_descriptor(another_agent.as_function_descriptor())


pipeline = qa_prompt >> agent

print(pipeline.get_required_context_keys())
print(pipeline.run(city="Berlin"))

```

### RAG

At the moment, only the simplest form of RAG is supported.

```python
from agent_dingo.rag.chunkers.recursive import RecursiveChunker
from agent_dingo.rag.embedders.sentence_transformer import SentenceTransformer
from agent_dingo.rag.vector_stores.chromadb import ChromaDB
from agent_dingo.rag.readers.list import ListReader
from agent_dingo.rag.prompt_modifiers import RAGPromptModifier
from agent_dingo.core.blocks import ChatPrompt
from agent_dingo.core.message import UserMessage

### build
text = """
...
"""
# Initialize reader, chunker, embedder and vector store
reader = ListReader()
chunker = RecursiveChunker(chunk_size=256)
embedder = SentenceTransformer()
vs = ChromaDB(collection_name="test", path="./here", recreate_collection=True)
# prepare the vector store
chunks = chunker.chunk(reader.read([text]))
embedder.embed_chunks(chunks)
vs.upsert_chunks(chunks)

### run
prompt = ChatPrompt([UserMessage("...")])
llm = OpenAI(model="gpt-3.5-turbo")
rag = RAGPromptModifier(embedder, vs)
pipeline = rag >> llm

pipeline.run(prompt)
```

### Web Server

The pipeline can be invoked using a REST API. The server can be started using the `serve_pipeline` function. The function takes either a single pipeline (which will be used as a default model and named "dingo") or a dictionary of pipelines (where the keys are the model names).

```python
from agent_dingo.core.state import State, ChatPrompt, KVData
from agent_dingo.core.blocks import *
from agent_dingo.core.message import *
from agent_dingo.serve import serve_pipeline

from agent_dingo.llm.openai import OpenAI

llm = OpenAI(model="gpt-3.5-turbo")

translation_prompt_template = "Translate to {language}: {text}"

translation_builder = PromptBuilder(
    [UserMessage(translation_prompt_template)], from_state=["text"]
)

pipeline = llm >> translation_builder >> llm

pipeline_raw = llm.as_pipeline()

serve_pipeline({"gpt35-translated": pipeline, "gpt35-raw": pipeline_raw}, port = 8000, is_async=True)

```

Once the server is running, the pipeline can be accessed using the `openai` python package:

```python
import openai

client = openai.OpenAI(base_url="http://localhost:8000")

messages = [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "context_language", "content": "French"}, # context keys are passed as custom roles with the "context_" prefix
]

out = client.chat.completions.create(messages=messages, model="gpt35-translated")

print(out)

out = client.chat.completions.create(messages=messages, model="gpt35-raw") # the language context will be ignored

print(out)

models = client.models.list()
print(models.models[0])

```
