from __future__ import annotations

import abc
from typing import List, TypeVar, Generic, Tuple, Optional, Generator, Type, Dict, Any

import pydantic
from datasets import load_dataset

from agents import StudentSeed

T = TypeVar('T')  # Input or current type
U = TypeVar('U')  # Output or next type


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


class Emitter(Generic[T, U]):

    def __init__(
            self,
            stage: Optional['Stage[T, U]'],
            next_emitter: Optional['Emitter[U]'],
            tracker: Dict[str, int]
    ):

        if (stage is None) != (next_emitter is None):
            raise ValueError("Must specify both next_stage and next_emitter or neither.")

        self._stage = stage
        self.next_emitter = next_emitter
        self._tracker = tracker

    def emit(self, sample: T) -> None:
        if self._stage and self.next_emitter:
            self._stage.process(sample, self.next_emitter)

    def add(self, name: str) -> None:
        self._tracker[name] = self._tracker.get(name, 0) + 1


class Stage(Generic[T, U], abc.ABC):

    @abc.abstractmethod
    def process(self, sample: T, emitter: Emitter[U]) -> None:
        ...

    def cleanup(self, emitter: Emitter[U]) -> None:
        ...


class DataSource(Generic[T], abc.ABC):

    @abc.abstractmethod
    def read(self) -> Generator[T, None, None]:
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


class BufferStage(Stage[T, List[T]]):

    def __init__(self):
        self._buffer: List[T] = []

    def process(self, sample: T, emitter: Emitter[List[T]]) -> None:
        self._buffer.append(sample)

    def cleanup(self, emitter: Emitter[List[T]]) -> None:
        emitter.emit(self._buffer)
        self._buffer.clear()


# class GenSeed()


class CollectSink(Stage[T, None]):
    def __init__(self):
        self.items: list[T] = []

    def process(self, sample: T, emitter: Emitter[None]) -> None:
        self.items.append(sample)


class PipelineStep(Generic[T, U]):
    def __init__(self, previous: Optional['PipelineStep[Any, U]'], stage: Optional[Stage[T, U]]):
        self.previous = previous
        self.stage = stage


class SocraticBench(Generic[T]):

    def __init__(self, source: DataSource[T], step: Optional[PipelineStep[T, U]] = None):
        self._source = source
        self._last_step = step

    @classmethod
    def default(cls) -> SocraticBench[Record]:
        ...

    @classmethod
    def from_data(cls: Type[SocraticBench[T]], source: DataSource[T]) -> SocraticBench[T]:
        return cls(source)

    def apply(self, stage: Stage[T, U]) -> 'SocraticBench[U]':
        last_step = PipelineStep(self._last_step, stage)
        return SocraticBench(self._source, last_step)

    def batch(self, stage: Stage[List[T], U]) -> 'SocraticBench[U]':
        buffered: BufferStage[T] = BufferStage()
        return self.apply(buffered).apply(stage)

    def run(self) -> Tuple[List[T], Tracker]:
        tracker: Dict[str, int] = {}
        sink: CollectSink[T] = CollectSink()
        terminal: Emitter[None, None] = Emitter(None, None, tracker)
        final_sink: Emitter[None, T] = Emitter(sink, terminal, tracker)
        current_emitter = final_sink

        step = self._last_step
        while step and step.stage is not None:
            current_emitter = Emitter(step.stage, current_emitter, tracker)
            step = step.previous

        for sample in self._source.read():
            current_emitter.emit(sample)

        return sink.items, tracker


### TMP Classes

class Tokenize(Stage[str, list[str]]):
    def process(self, sample: str, emitter: Emitter[list[str]]) -> None:
        emitter.emit(sample.split())


class Count(Stage[list[str], int]):
    def process(self, sample: list[str], emitter: Emitter[int]) -> None:
        emitter.emit(len(sample))


class StringSource(DataSource[str]):
    def __init__(self, data: list[str]):
        self._data = data

    def read(self) -> Generator[str, None, None]:
        yield from self._data


if __name__ == "__main__":
    strsource: DataSource[str] = StringSource(["hello world", "this is SocraticBench"])

    pipeline = (
        SocraticBench.from_data(strsource)
        .apply(Tokenize())  # str → list[str]
        .apply(Count())  # list[str] → int
    )
    items, t = pipeline.run()

    bench = SocraticBench.from_data(strsource)  # type: SocraticBench[str]
    # b2 = bench.apply(Tokenize())
    # b3 = b2.apply(Count())
    print(items)

    # API - pipeline
    # s = SocraticBench.default()
    # s.run()

    # API - selecting pipeline stages
    # s = (SocraticBench.from_data(PrincetonChapters(Record, num_conversations=10))
    #      .with_stage(SeedStage(None))
    #      .with_state()
    #      # .with_stage(DumpStage())
    #      # .with_stage(DumpStage())
    #      )
    # s.run()

    # API - fine control (simulating single chat)

    # API - overriding behavior

    # API - using local models (HF transformers)
    pass
