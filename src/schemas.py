from typing import Optional, List

import numpy as np
from pydantic import BaseModel, RootModel


class Seed(BaseModel):
    source_content: str
    question: str
    main_topics: str
    interaction_type: str

    def __str__(self) -> str:
        return (f"# Source content\n{self.source_content}\n\n"
                f"# Main Topics\n{self.main_topics}\n\n"
                f"# Interaction type\n{self.interaction_type}\n\n"
                f"# Question\n{self.question}")


class SeedDataset(RootModel):
    root: list[Seed]

    def __str__(self) -> str:
        return "\n\n-------\n\n".join(str(s) for s in self.root)


class Message(BaseModel):
    role: str
    content: str
    end: bool

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"


class ChatHistory(RootModel):
    root: list[Message]

    def __str__(self) -> str:
        return "\n".join(str(m) for m in self.root)

    def __len__(self) -> int:
        return len(self.root)


class Interaction(BaseModel):
    seed: Seed
    student_type: str
    chat_history: ChatHistory

    def __str__(self) -> str:
        return f"{self.seed}\n\n# Student type\n{self.student_type}\n\n# Chat History\n{self.chat_history}"


class InteractionMetadata(BaseModel):
    student_llm: str
    teacher_llm: str
    max_interactions: int


class InteractionDataset(BaseModel):
    metadata: InteractionMetadata
    interactions: List[Interaction]

    def __str__(self) -> str:
        return "\n\n-------\n\n".join(str(i) for i in self.interactions)


class Evaluation(BaseModel):
    id: int
    interaction: Interaction
    feedback: Optional[str]
    assessment: Optional[bool]

    def __str__(self) -> str:
        return f"{self.interaction}\n\n# Feedback\n{self.feedback}\n\n# Assessment\n{self.assessment}"


class EvalMetadata(BaseModel):
    student_llm: str
    judge_llm: str
    teacher_llm: str
    max_interactions: int


class EvaluationDataset(BaseModel):
    metadata: EvalMetadata
    evaluations: List[Evaluation]

    def __str__(self) -> str:
        return "\n\n-------\n\n".join(str(e) for e in self.evaluations)

    def __len__(self) -> int:
        return len(self.evaluations)

    def get_valid(self) -> List[Evaluation]:
        return [e for e in self.evaluations if e.assessment is not None]

    def avg_performance(self) -> float:
        return np.mean([e.assessment for e in self.get_valid()])


class JudgeDataset(BaseModel):
    model_name: str
    evaluations: list[Evaluation]

    def avg_performance(self) -> float:
        return np.mean([e.assessment for e in self.evaluations])
