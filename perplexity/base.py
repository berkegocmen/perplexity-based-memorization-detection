import weave
from pydantic import BaseModel


class Prompt(BaseModel):
    pass

class GeneratedCode(BaseModel):
    prompt: str
    generated_code: str

    # complete code as prompt + generated code
    @property
    def complete_code(self):
        return self.prompt + self.generated_code




