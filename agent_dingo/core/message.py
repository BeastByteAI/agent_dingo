class Message:
    role: str = "undefined"

    def __init__(self, content: str):
        self.content = content

    def __repr__(self):
        return f'Message(role="{self.role}" content="{self.content}")'

    @property
    def dict(self):
        return {"role": self.role, "content": self.content}


class UserMessage(Message):
    role = "user"
    pass


class SystemMessage(Message):
    role = "system"
    pass


class AssistantMessage(Message):
    role = "assistant"
    pass
