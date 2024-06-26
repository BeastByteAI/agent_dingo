from agent_dingo.agent.function_descriptor import FunctionDescriptor
from langchain.tools import (
    BaseTool as _BaseLangchainTool,
    format_tool_to_openai_function as _format_tool_to_openai_function,
)


def convert_langchain_tool(
    tool: _BaseLangchainTool, make_async: bool = False
) -> FunctionDescriptor:
    """Converts a langchain tool to a function descriptor.

    Parameters
    ----------
    tool : _BaseLangchainTool
        The langchain tool.

    Returns
    -------
    FunctionDescriptor
        The function descriptor.
    """
    if not isinstance(tool, _BaseLangchainTool):
        raise ValueError("tool must be a subclass of langchain.tools.BaseTool")

    json_repr = _format_tool_to_openai_function(tool=tool)
    name = json_repr["name"]
    if make_async:

        async def func(__arg1):
            return await tool.arun(tool_input=__arg1)

    else:

        def func(__arg1):
            return tool.run(tool_input=__arg1)

    requires_context = False
    descriptor = FunctionDescriptor(
        name=name, func=func, json_repr=json_repr, requires_context=requires_context
    )
    return descriptor
