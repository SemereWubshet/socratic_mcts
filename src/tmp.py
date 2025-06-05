import abc
from typing import List, TypeVar, Generic, Tuple, Optional, Generator, Type

import pydantic
from datasets import load_dataset

T = TypeVar('T', bound='Record')


class Metadata(pydantic.BaseModel):
    student_llm: Optional[str] = None
    teacher_llm: Optional[str] = None
    judge_llm: Optional[str] = None
    max_interactions: Optional[int] = None


class Seed(pydantic.BaseModel):
    source_content: Optional[str] = None
    question: Optional[str] = None
    main_topics: Optional[str] = None
    interaction_type: Optional[str] = None


class Message(pydantic.BaseModel):
    role: str
    content: str
    end: bool

    def __str__(self) -> str:
        return f"{self.role}: {self.content}"


class ChatHistory(pydantic.RootModel):
    root: list[Message]

    def __str__(self) -> str:
        return "\n".join(str(m) for m in self.root)

    def __len__(self) -> int:
        return len(self.root)


class Record(pydantic.BaseModel):
    metadata: Metadata = Metadata()
    id: int
    seed: Seed = Seed()
    student_type: Optional[str] = None
    chat_history: Optional[ChatHistory] = None
    feedback: Optional[str] = None
    assessment: Optional[bool] = None


class Tracker(abc.ABC):
    pass


class Stage(Generic[T], abc.ABC):

    @abc.abstractmethod
    def process(self, sample: T, tracker: Tracker) -> None:
        ...


class DataSource(Generic[T], abc.ABC):

    @abc.abstractmethod
    def read(self) -> Generator[Tuple[str, T]]:
        ...


class PrincetonChapters(Generic[T], DataSource[T]):

    def __init__(
            self,
            record_cls: Type[T],
            num_conversations: int,
            max_chapter_size: int = 3000,
    ):
        self._record_cls = record_cls
        self._num_conversations = num_conversations
        self._max_chapter_size = max_chapter_size

    def read(self) -> Generator[T, None, None]:
        chapters = load_dataset("princeton-nlp/TextbookChapters", split="train")
        chapters = chapters.shuffle()

        i = 0
        while i < self._num_conversations and i < len(chapters):
            page = chapters[i]
            text: str = page["chapter"][:self._max_chapter_size]

            record = self._record_cls(
                id=i,
                seed=Seed(source_content=text),
            )
            yield record
            i += 1


# class GenSeed()


class DumpStage(Stage[T, T]):

    def process(self, sample: T, tracker: Tracker) -> None:
        pass


class SocraticBench(Generic[T]):

    def __init__(self):
        ...

    def from_data(self, source: DataSource) -> 'SocraticBench[T]':
        ...

    def with_stage(self, step: Stage[T]) -> 'SocraticBench[T]':
        ...

    def branch(self, stages: List[Stage[T]]) -> 'SocraticBench[T]':
        ...

    def run(self) -> Tuple[List[T], Tracker]:
        ...

    @classmethod
    def default(cls) -> 'SocraticBench[Record]':
        ...


if __name__ == "__main__":
    # API - pipeline
    s = SocraticBench.default()
    s.run()

    # API - selecting pipeline stages
    s = (SocraticBench()
         .from_data(PrincetonChapters(Record, num_conversations=10))
         .with_stage(DumpStage())
         .with_stage(DumpStage()))
    s.run()

    # API - fine control (simulating single chat)

    # API - overriding behavior

    # API - using local models (HF transformers)
    pass
