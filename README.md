<h1 align="center">
  <br>
 <img src="https://gist.githubusercontent.com/OKUA1/55e2fb9dd55673ec05281e0247de6202/raw/41063fcd620d9091662fc6473f9331a7651b4465/dingo.svg" alt="AgentDingo" width="250" height = "250">
  <br>
  Agent Dingo
  <br>
</h1>

<h4 align="center">A microframework for building LLM-powered pipelines and agents.</h4>

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

## Support us 🤝

You can support the project in the following ways:

- ⭐ Star Dingo on GitHub (click the star button in the top right corner)
- 💡 Provide your feedback or propose ideas in the [issues](https://github.com/BeastByteAI/agent_dingo/issues) section or [Discord](https://discord.gg/YDAbwuWK7V)
- 📰 Post about Dingo on LinkedIn or other platforms
- 🔗 Check out our other projects: <a href="https://github.com/iryna-kondr/scikit-llm">Scikit-LLM</a>, <a href="https://github.com/beastbyteai/falcon">Falcon</a>

<br>
<a href="https://github.com/iryna-kondr/scikit-llm">
  <picture>
  <source media="(prefers-color-scheme: light)" srcset="https://gist.githubusercontent.com/OKUA1/ce2167df8e441ce34a9fbc8578b86543/raw/f740c391ec37eaf2f80d5b46f1fa2a989dd45932/skll_h_dark.svg" >
  <source media="(prefers-color-scheme: dark)" srcset="https://gist.githubusercontent.com/OKUA1/ce2167df8e441ce34a9fbc8578b86543/raw/f740c391ec37eaf2f80d5b46f1fa2a989dd45932/skllm_h_light.svg">
  <img alt="Logo" src="https://gist.githubusercontent.com/OKUA1/ce2167df8e441ce34a9fbc8578b86543/raw/f740c391ec37eaf2f80d5b46f1fa2a989dd45932/skll_h_dark.svg" height = "65">
</picture>
</a> <br><br>
<a href="https://github.com/OKUA1/falcon">
  <picture>
  <source media="(prefers-color-scheme: light)" srcset="https://gist.githubusercontent.com/OKUA1/ce2167df8e441ce34a9fbc8578b86543/raw/f740c391ec37eaf2f80d5b46f1fa2a989dd45932/falcon_h_dark.svg" >
  <source media="(prefers-color-scheme: dark)" srcset="https://gist.githubusercontent.com/OKUA1/ce2167df8e441ce34a9fbc8578b86543/raw/f740c391ec37eaf2f80d5b46f1fa2a989dd45932/falcon_h_light.svg">
  <img alt="Logo" src="https://gist.githubusercontent.com/OKUA1/ce2167df8e441ce34a9fbc8578b86543/raw/f740c391ec37eaf2f80d5b46f1fa2a989dd45932/dingo_h_dark.svg" height = "65">
</picture>
</a>

## Quick Start & Documentation 🚀

**Step 1:** Install `agent-dingo`

```bash
pip install agent-dingo
```

**Step 2:** Configure your OpenAI API key

```bash
export OPENAI_API_KEY=<YOUR_KEY>
```

**Step 3:** Build your pipeline

Example 1 (Linear Pipeline):

````python
from agent_dingo.llm.openai import OpenAI
from agent_dingo.core.blocks import PromptBuilder
from agent_dingo.core.message import UserMessage
from agent_dingo.core.state import ChatPrompt


# Model
gpt = OpenAI("gpt-3.5-turbo")

# Summary prompt block
summary_pb = PromptBuilder(
    [UserMessage("Summarize the text in 10 words: ```{text}```.")]
)

# Translation prompt block
translation_pb = PromptBuilder(
    [UserMessage("Translate the text into {language}: ```{summarized_text}```.")],
    from_state=["summarized_text"],
)

# Pipeline
pipeline = summary_pb >> gpt >> translation_pb >> gpt

input_text = """
Dingo is an ancient lineage of dog found in Australia, exhibiting a lean and sturdy physique adapted for speed and endurance, dingoes feature a wedge-shaped skull and come in colorations like light ginger, black and tan, or creamy white. They share a close genetic relationship with the New Guinea singing dog, diverging early from the domestic dog lineage. Dingoes typically form packs composed of a mated pair and their offspring, indicating social structures that have persisted through their history, dating back approximately 3,500 years in Australia.
"""

output = pipeline.run(text = input_text, language = "french")
print(output)
````

Example 2 (Agent):

```python
from agent_dingo.agent import Agent
from agent_dingo.llm.openai import OpenAI
import requests

llm = OpenAI(model="gpt-3.5-turbo")
agent = Agent(llm, max_function_calls=3)

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

pipeline = agent.as_pipeline()
```

For a more detailed overview and additional examples, please refer to the **[documentation](https://dingo.beastbyte.ai/)**.
