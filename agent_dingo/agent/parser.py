from docstring_parser import parse as _parse
import ast


_types = {
    "str": "string",
    "int": "integer",
    "float": "number",
    "bool": "boolean",
    "list": "array",
    "dict": "object",
}


def parse(docstring: str) -> dict:
    """Parses a docstring.

    Parameters
    ----------
    docstring : str
        The docstring to parse.

    Returns
    -------
    dict
        A dictionary containing the description and the arguments of the function.

    Raises
    ------
    ValueError
        If the docstring has no description.
    """
    parsed = _parse(docstring)
    description = ""
    if parsed.short_description:
        description = parsed.short_description
    if parsed.long_description:
        if description != "":
            description += "\n" + parsed.long_description
        else:
            description = parsed.long_description
    if description == "":
        raise ValueError("Docstring has no description")
    args = {}
    requires_context = False
    for arg in parsed.params:
        if arg.arg_name == "chat_context" and arg.type_name == "ChatContext":
            requires_context = True
            continue
        d = {}
        if "Enum:" in arg.description:
            arg_description = arg.description.split("Enum:")[0].strip()
            enum = arg.description.split("Enum:")[1].strip()
            try:
                enum = ast.literal_eval(enum)
                d = {"description": arg_description, "enum": enum}
            except Exception:
                d = {"description": arg_description}
        else:
            d = {"description": arg.description}
        if arg.type_name in _types:
            args[arg.arg_name] = {"type": _types[arg.type_name]}
        else:
            args[arg.arg_name] = {"type": "string"}
        args[arg.arg_name].update(d)
    return {"description": description, "properties": args}, requires_context
