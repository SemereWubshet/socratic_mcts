import abc
from typing import List, TypeVar, Generic, Tuple, Optional, Generator, Type, Dict

import pydantic
from datasets import load_dataset

from agents import StudentSeed

T = TypeVar('T', bound='Record')

I = TypeVar('I')
O = TypeVar('O')


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


class Emitter(Generic[O]):

    def __init__(self, next_stage: 'Stage', tracker: Dict[str, int]):
        self._next_stage = next_stage
        self._tracker = tracker

    def emit(self, sample: O) -> None:
        self._next_stage.process(sample, self)

    def add(self, name: str) -> None:
        c = self._tracker.get(name, 0)
        self._tracker[name] = c + 1


class Stage(Generic[I, O], abc.ABC):

    @abc.abstractmethod
    def process(self, sample: I, emitter: Emitter[O]) -> None:
        ...

    def cleanup(self, emitter: Emitter[O]) -> None:
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


class SeedStage(Stage[str, Seed]):

    def __init__(self, seed: StudentSeed):
        self._pool = []
        self._seed = seed  # TODO: use interface

    def process(self, sample: str, emitter: Emitter[List[Seed]]) -> None:
        question, topics = self._seed.gen_seed(sample)
        # TODO: interaction type
        seed = Seed(source_content=sample, question=question, main_topics=topics, interaction_type=0)
        self._pool.append(seed)

    def cleanup(self, emitter: Emitter[List[Seed]]) -> None:
        pass


class BufferStage(Stage[I, List[I]]):

    def __init__(self):
        self._buffer: List[I] = []

    def process(self, sample: I, emitter: Emitter[I]) -> None:
        self._buffer.append(sample)

    def cleanup(self, emitter: Emitter[I]) -> None:
        emitter.emit(self._buffer)
        self._buffer.clear()


# class GenSeed()


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
         # .with_stage(DumpStage())
         # .with_stage(DumpStage())
         )
    s.run()

    # API - fine control (simulating single chat)

    # API - overriding behavior

    # API - using local models (HF transformers)
    pass
