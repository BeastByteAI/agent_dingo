<h1 align="center">
  <br>
 <img src="https://github.com/OKUA1/agent_dingo/blob/main/logo.png?raw=true" alt="AgentDingo" width="250" height = "250">
  <br>
  Agent Dingo
  <br>
</h1>

<h4 align="center">A microframework for building LLM-powered applications.</h4>

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

_Dingo_ is a compact LLM orchestration framework designed for straightforward development of production-ready LLM-powered applications. It combines simplicity with flexibility, allowing for the efficient construction of pipelines and agents, while maintaining a high level of control over the process.

## Quick Start ⚡️
> ⚠️ Currently stable version is `v0.1`, which is going to be replaced soon by `v1` that introduces many new features like pipelines, RAG, etc. The most recent (unstable) version can be found in [this branch](https://github.com/BeastByteAI/agent_dingo/tree/pipelines).

**Step 1:** Install `agent-dingo`

```bash
pip install agent-dingo
```

**Step 2:** Configure your OpenAI API key

```bash
export OPENAI_API_KEY=<YOUR_KEY>
```

**Step 3:** Instantiate the agent

```python
from agent_dingo import AgentDingo

agent = AgentDingo()
```

**Step 4:** Add `agent.function` decorator to the function you wish to integrate

```python
@agent.function
def get_current_weather(city: str):
    ...
```

**Step 5:** Run the conversation

```python
agent.chat("What is the current weather in Linz?")
```

**Optional:** Run an OpenAI compatible server

```python
from agent_dingo.wrapper import DingoWrapper
DingoWrapper(agent).serve()
```

The server can be accessed using the `openai` python package:

```python
import openai

openai.api_base = "http://localhost:8080"

r = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages = [{"role": "user", "content": "What is the current weather in Linz?"}],
    temperature=0.0,
)
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

### OpenAI API Key

_Dingo_ is built around function calling feature of newer generation OpenAI chat models that were explicitly fine-tuned for these tasks.
Hence, an OpenAI key is required.

You can either set the `OPENAI_API_KEY` env variable or register the key using the `openai` python package.

```bash
export OPENAI_API_KEY=<YOUR_KEY>
```

```python
import openai

openai.api_key = "<YOUR_KEY>"
```

### Agent

`AgentDingo` is a central part of the framework which allows you to register the functions to use with ChatGPT. The intermediate function calling is also handled by the agent directly.

```python
from agent_dingo import AgentDingo

agent = AgentDingo()
```

### Registering the functions

**Option 1** (Recommended): Registering the function with a docstring

By default, the agent uses the information from the docstring to generate a function descriptor that is passed to the model. It is advised to always use the functions with docstrings as this way you can describe the purpose of the function (and its arguments) more accurately. The library was explicitly tested with `google` and `numpy` docstring styles.

Example:

```python
@agent.function
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
```

**Option 2**: Registering the function without a docstring

If the function does not have a docstring, it can still be registered. In that case, the docstring will be generated automatically by ChatGPT.
However, there are several drawbacks of this approach:

- The source code of your function is passed to the model;
- The generated docstring might be inaccurate (especially for complex functions);
- The generated docstrings are not persistant (i.e. they are not saved to disk) and will have to be re-generated every time the function is registered.

If you wish to disable the automatic docstring generation, you can set the `allow_codegen` parameter to `False` when instantiating the agent.

Example:

```python
agent = AgentDingo(allow_codegen=False)
```

By default the `allow_codegen` parameter is set to `"env"` which means that the value is read from the `DINGO_ALLOW_CODEGEN` environment variable. If the variable is not set, it is assumed to be `True`.

It is also possible to change the model used for the code generation by setting the `DINGO_CODEGEN_MODEL` environment variable. By default, the `gpt-3.5-turbo-0613` model is used.

```bash
export DINGO_CODEGEN_MODEL="gpt-4-0613"
```

### Registering of external functions

If you wish to register a function that is not defined in the current file, you can use the `register_function` method of the agent.

```python
from my_module import get_temperature

agent.register_function(get_temperature)
```

Alternatively, you can define a function descriptor manually and register it using the `register_descriptor` method. In this case, a `json_representation` compatible with [OpenAI function calling API](https://platform.openai.com/docs/api-reference/chat/create#chat/create-functions) should be provided.

```python
from agent_dingo.descriptor import FunctionDescriptor

d = FunctionDescriptor(
  name = "<function_name>",
  json_representation = {name: "<function_name>", description: "<function_description>", parameters: ...}
  func = function_callable
  requires_context = True or False
)

agent.register_descriptor(d)
```

### Running the conversation

Once the functions are registered, you can run the conversation using the `chat` method of the agent.

```python
agent.chat("What is the current temperature in Linz?")
```

The chat method accepts the following parameters:

- `messages` - the message to start the conversation with; it can either be a string or a list of messages (conversation history);
- `chat_context` - the global context of the conversation (more details are provided in the next section); by default, it is set to `None`;
- `model` - the model to use for the conversation; by default, the `gpt-3.5-turbo-0613` model is used;
- `temperature` - the randomness parameter of the model; by default, it is set to `1.0`;
- `max_function_calls` - the maximum number of function calls allowed during the conversation; by default, it is set to `10`;
- `before_function_call` - an interceptor that is called before the function is called (mode details are provided in the next section); by default, it is set to `None`.

All of the parameters except `messages` are optional.

The method returns a tuple which contains the last message of the conversation (as string) and the full conversation history (including function calls).

### Chat context

In some cases the function might require to access the global context of the conversation. For example, the function might need to access some user-specific information (e.g. user id). In this case, the `chat_context` parameter can be used. It is a special dictionary that is passed to the function and can contain any information that is required. Unlike other arguments, the content of the chat_context is not generated by the model and is passed directly to the function.

```python
from agent_dingo.context import ChatContext

@agent.function
def get_user_name(greeting_msg: str, chat_context: ChatContext) -> str:
    """Returns a greeting message with the user's name.

    Parameters
    ----------
    greeting_msg : str
        Message to greet the user with.
    chat_context : ChatContext
        The chat context.

    Returns
    -------
    str
        The greeting message with the user's name.
    """
    user_name = chat_context["user_name"]
    return f"{greeting_msg}, {user_name}!"

r = agent.chat(
    "Say hi.", chat_context=ChatContext(user_name="John"), temperature=0.0
)

# > Hi, John! How can I assist you today?
```

Note: the `chat_context` parameter is not passed to the model and is not used for the generation of the function descriptor.

### Before function call interceptor

In some cases, it might be required to perform some actions before the function is called. For example, you might want to log the function call or perform some checks. This is especially handy since the function arguments generated by the model are not guaranteed to be correct/valid, hence, it is advised to add some additional validators. The `before_function_call` parameter can be used to register an interceptor that is called before the function. The interceptor receives the following parameters: `function_name`, `function_callable`, `function_kwargs` and should return a tuple with the updated `function_callable` and `function_kwargs`.

Example: intercepting the function call and logging the function name and arguments.

```python
def before_function_call(function_name: str, function_callable: Callable, function_kwargs: dict):
    print(f"Calling function {function_name} with arguments {function_kwargs}")
    return function_callable, function_kwargs

agent.chat(
    "What is the current temperature in Linz?",
    before_function_call=before_function_call,
)
```

### DingoWrapper + Web Server

In addition to using the agent directly, it is possible to wrap it into a `DingoWrapper`, which provides an OpenAI-like API.

```python
from agent_dingo.wrapper import DingoWrapper
wrapped_agent = DingoWrapper(agent, before_function_call = None, max_function_calls=10)
```

Once the agent is wrapped, it can be used to create the chat completions using the `chat_completion` method.

```python
r = wrapped_agent.chat_completion(
    messages = [{"role": "user", "content": "What is the current weather in Linz?"}],
    model = "gpt-3.5-turbo",
    temperature=0.0, #optional
    chat_context=None, #optional
)
```

In principle, this method can be used as a drop-in replacement for the `openai.ChatCompletion.create` method. However, there are several differences:

- DingoWrapper does not support most of the optional hyperparameters of the `openai.ChatCompletion.create` method (except `temperature`);
- DingoWrapper has an additional (optional) `chat_context` parameter that can be used to pass the global context of the conversation;

Example:

```python
# openai.ChatCompletion
r = openai.ChatCompletion.create(
    messages = [{"role": "user", "content": "What is the current weather in Linz?"}],
    model = "gpt-3.5-turbo",
    temperature=0.0
)

# DingoWrapper
r = wrapped_agent.chat_completion(
    messages = [{"role": "user", "content": "What is the current weather in Linz?"}],
    model = "gpt-3.5-turbo",
    temperature=0.0
)
```

The `DingoWrapper` can also be used to run a web server (also compatible with the OpenAI API). The server can be started using the `serve` method.

The serve method requires additional dependencies:

```bash
pip install agent_dingo[server]
```

```python
wrapped_agent.serve(port=8080, host="0.0.0.0", threads=4)
```

Once the server has started, it can be accessed using e.g. the `openai` python package.

```python
# client.py
import openai

openai.api_base = "http://localhost:8080"

r = openai.ChatCompletion.create(
    model = "gpt-3.5-turbo",
    messages = [{"role": "user", "content": "What is the temperature in Linz ?"}],
    temperature=0.0,
)

print(r)
```

Response:

```json
{
  "choices": [
    {
      "finish_reason": "stop",
      "index": 0,
      "message": {
        "content": "The current temperature in Linz is 25\u00b0C and it is sunny.",
        "role": "assistant"
      }
    }
  ],
  "created": 1692537919,
  "id": "chatcmpl-d6a9d6cc-7a26-41d5-a4a6-2c737b652f4b",
  "model": "dingo-gpt-3.5-turbo-0613",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 32,
    "prompt_tokens": 318,
    "total_tokens": 350
  }
}
```

_Note: the "usage" metric accumulates the number of tokens used for all intermediate function calls during the conversation._

### LangChain Tools 🦜️🔗

It is possible to convert [Langchain Tools](https://python.langchain.com/docs/modules/agents/tools/) into function descriptors in order to register them with Dingo. The converter can be used as follows:

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

Please refer to the [LangChain documentation](https://api.python.langchain.com/en/latest/api_reference.html#module-langchain.tools) for more details on how to define the tools.

3. Convert the tool into a function descriptor and register:

```python
from agent_dingo.langchain import convert_langchain_tool
descriptor = convert_langchain_tool(tool)
agent.register_descriptor(descriptor)
```

4. Run the conversation:

```python
# The agent will query Wikipedia to obtain the answer.
agent.chat("What is LangChain according to Wikipedia? Explain in one sentence.")

# > According to Wikipedia, LangChain is a framework designed to simplify the creation of applications using large language models (LLMs), with use-cases including document analysis and summarization, chatbots, and code analysis.
```

In comparison, when we try to query ChatGPT directly with the same question, we get the following hallucinated response (since it does not have access to the relevant up-to-date information):

```python
# > LangChain is a blockchain-based platform that aims to provide language learning services and connect language learners with native speakers for real-time practice and feedback.
```

_Note: some of the tools might be incompatible with (or simply unsuitable for) Dingo. We do not guarantee that all of the tools will work out of the box._
