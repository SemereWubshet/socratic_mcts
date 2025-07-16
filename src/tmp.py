from __future__ import annotations

import abc
import random
import re
from typing import List, TypeVar, Generic, Tuple, Optional, Generator, Type, Dict, Any

import pydantic
from datasets import load_dataset

T = TypeVar('T')  # Input or current type
U = TypeVar('U')  # Output or next type


class LLMProcessingFailure(Exception):
    ...


class LLM(abc.ABC):

    @abc.abstractmethod
    def query(self, messages: List[Dict[str, str]]) -> str:
        ...

    @abc.abstractmethod
    def healthcheck(self) -> None:
        ...

    @property
    @abc.abstractmethod
    def model_name(self) -> str:
        ...

    def unload(self) -> None:
        pass


class ConversationSeeder(abc.ABC):

    @abc.abstractmethod
    def gen_seed(self, source_content: str, **kwargs: Dict[str, str]) -> Tuple[str, str]:
        ...

    @abc.abstractmethod
    def base_prompt(self) -> str:
        ...

    @abc.abstractmethod
    def interaction_types(self) -> Tuple[Dict[str, str]]:
        ...


class ContentSeeder(ConversationSeeder):

    def __init__(self, llm: LLM, max_trials: int = 10):
        """
        Args:
            llm: The language model used for seed generation.
            max_trials: Number of attempts to retry on malformed LLM output.
        """
        self._llm = llm
        self._max_trials = max_trials

    def base_prompt(self) -> str:
        return (
            "# Instructions\n"
            "You are a student trying to gain more understanding on a class topic. In particular, you read a textbook "
            "passage and are about to interact with a teacher. Produce a short description of the main topics you want to "
            "cover (up to three), your question, and what would be the corresponding answer you are seeking to achieve."
            "\n"
            "The question must be short, concise and hint about the main topics, but without disclosing what are the main "
            "topic to the teacher. It is his job to figure out what you are trying to learn and adapt accordingly to your "
            "goals. {interaction_type}"
            "\n"
            "The question must be understandable on its own because the teacher does not have access to the textbook "
            "passage you read."
            "\n\n"
            "# Output Format\n"
            "Your evaluation must have the format [MAIN_TOPICS]A description on what are the main topics you are seeking "
            "to learn - up to five points[\MAIN_TOPICS]The opening question[\QUESTION]. Do not output opening or closing "
            "statements or any special formatting."
            "\n\n"
            "# Example\n"
            "```\n"
            "{context}\n"
            "```"
            "\n"
            "OUTPUT: [MAIN_TOPICS]{main_topics}[\MAIN_TOPICS]{question}[\QUESTION]"
        )

    def interaction_types(self) -> Tuple[Dict[str, str]]:
        return (
            {
                "interaction_type": "Ask a general question about the main topic.",
                "context": "Rayleigh scattering is the phenomenon where light or other electromagnetic radiation is "
                           "scattered by particles much smaller than the wavelength of the light, typically molecules in "
                           "the atmosphere. This scattering is more effective at shorter wavelengths, meaning colors like "
                           "blue and violet are scattered more than longer wavelengths like red. This is why the sky "
                           "appears blue during the day. The intensity of Rayleigh scattering is inversely proportional to "
                           "the fourth power of the wavelength, which explains why shorter wavelengths are scattered much "
                           "more efficiently.",
                "main_topics": "- Scattering of light by particles smaller than the light's wavelength.\\n"
                               "- Shorter wavelengths are scattered more than longer wavelengths.\\n"
                               "- Scattering intensity is inversely proportional to the fourth power of the wavelength.\\n"
                               "- Role of molecules in the atmosphere in scattering light.",
                "question": "Why is the sky blue?"
            },
            {
                "interaction_type": "Ask a misleading question about the topic containing a wrong claim.",
                "context": "Rayleigh scattering is the phenomenon where light or other electromagnetic radiation is "
                           "scattered by particles much smaller than the wavelength of the light, typically molecules in "
                           "the atmosphere. This scattering is more effective at shorter wavelengths, meaning colors like "
                           "blue and violet are scattered more than longer wavelengths like red. This is why the sky "
                           "appears blue during the day. The intensity of Rayleigh scattering is inversely proportional to "
                           "the fourth power of the wavelength, which explains why shorter wavelengths are scattered much "
                           "more efficiently.",
                "main_topics": "- Explanation of why the Sun appears orange/red during these times.\\n"
                               "- Increased scattering of shorter wavelengths (blue/violet) when sunlight travels through "
                               "a thicker atmosphere.\\n"
                               "- Addressing the misconception that air temperature directly affects light scattering.\\n"
                               "- How the longer atmospheric path at sunrise and sunset influences color perception.\\n"
                               "- Differences between Rayleigh scattering (molecules) and Mie scattering (larger "
                               "particles).",
                "question": "Is the sunrise orange because the Sun warms the air thus scattering the light?"
            }
        )

    def gen_seed(self, source_content: str, **kwargs: Dict[str, str]) -> Tuple[str, str]:
        system_prompt = self.base_prompt().format(**kwargs)
        trials = 0
        output = ""
        while trials < self._max_trials:
            output = self._llm.query(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"```\n{source_content}\n```\nOUTPUT: "}
                ]
            )
            output = output.strip()

            match = re.search(
                r"\[MAIN_TOPICS](?P<topics>.*?)\[\\MAIN_TOPICS](?P<question>.*?)\[\\QUESTION]", output, re.DOTALL
            )

            if match is None:
                trials += 1
                continue

            return match.group("question"), match.group("topics")

        raise LLMProcessingFailure(
            f"Failed getting LLM to output correct JSON for \n\n\n{source_content}\n\n\noutput: {output}"
        )


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

    def __init__(self, seeder: ConversationSeeder):
        self._seeder = seeder
        self._num_interactions = len(self._seeder.interaction_types())

    def process(self, sample: str, emitter: Emitter[Seed]) -> None:
        interaction_type = random.randint(0, self._num_interactions - 1)
        question, topics = self._seeder.gen_seed(sample, **self._seeder.interaction_types()[interaction_type])
        seed = Seed(source_content=sample, question=question, main_topics=topics, interaction_type=interaction_type)
        emitter.emit(seed)

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
