class UsageMeter:
    def __init__(self):
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.last_finish_reason = None

    def increment(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens

    def log_raw(self, completion_response: dict) -> None:
        usage = completion_response["usage"]
        self.increment(usage["prompt_tokens"], usage["completion_tokens"])
        self.last_finish_reason = completion_response["choices"][0]["finish_reason"]

    def get_usage(self) -> dict:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
        }
