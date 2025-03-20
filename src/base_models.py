from typing import Dict, List

from pydantic import RootModel, BaseModel


class Message(BaseModel):
    role: str
    content: str
    end: bool

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"

    def format(self) -> Dict[str, str]:
        return {"role": "user" if self.role.lower() == "student" else "assistant", "content": self.content}


class ChatHistory(RootModel):
    root: list[Message]

    def __str__(self) -> str:
        return "\n".join(str(m) for m in self.root)

    def format(self) -> List[Dict[str, str]]:
        return [m.format() for m in self.root]
