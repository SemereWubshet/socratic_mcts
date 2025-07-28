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
        ...


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

    def seed_llm(self) -> Optional[str]:
        return None


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
            f"Failed getting LLM to output correct for \n\n\n{source_content}\n\n\noutput: {output}"
        )

    def seed_llm(self) -> Optional[str]:
        return self._llm.model_name


class Student(abc.ABC):

    def __init__(self, llm: LLM):
        self._llm = llm

    @abc.abstractmethod
    def query(self, chat_history: ChatHistory, **kwargs: Dict[str, str]) -> Message:
        ...

    @abc.abstractmethod
    def system_prompt(self) -> str:
        ...

    @abc.abstractmethod
    def message_prompt(self) -> str:
        ...

    @abc.abstractmethod
    def student_types(self) -> Tuple[str]:
        ...

    def llm(self) -> LLM:
        return self._llm


class StudentAgent(Student):

    def __init__(self, llm: LLM, max_trials: int = 10):
        super().__init__(llm)
        self._max_trials = max_trials

    def system_prompt(self) -> str:
        return (
            "# Instructions\n"
            "\n"
            "{student_type}\n"
            "\n"
            "Continue the conversation with a teacher by making concise replies.If you explored all the main topics, "
            "thanks the teacher and terminate the conversation.Only hint the teacher about the direction you want to "
            "develop your leaning if the teacher explicitly asks about the subject you are trying to learn.Otherwise, "
            "reply to the teacher in a constructive way.\n"
            "\n"
            "# Output Format\n"
            "\n"
            "Your evaluation must start with a concise response to the teacher followed by the token[END] if you wish "
            "to stop the conversation or[CONTINUE] if you want to engage with the teacher for yet another round.Do not "
            "output opening or closing statements.\n"
            "\n"
            "# Examples\n"
            "\n"
            "# Main topics\n"
            "- Definition of Rayleigh Scattering\n"
            "- Wavelength Dependence\n"
            "- Atmospheric Molecules\n"
            "\n"
            "# Chat History\n"
            "Student: Why is the sky blue?\n"
            "Teacher: To begin, have you ever wondered what exactly we see when we look at the sky? What is it made "
            "of, and how does it interact with light?\n"
            "Student: The sky is made of molecules of mostly oxygen, nitrogen and carbon dioxide.\n"
            "Teacher: When sunlight reaches the Earth, it doesn’t just come as a single color, but as a mix of many "
            "colors.Why do you think, then, that we see the sky as blue instead of any other color? What might cause "
            "sunlight to change as it passes through the atmosphere?\n"
            "\n"
            "OUTPUT: Sunlite collision with air molecules changes their wavelengths?[CONTINUE]\n"
            "\n"
            "---\n"
            "\n"
            "# Main topics\n"
            "- Definition of Rayleigh Scattering\n"
            "- Wavelength Dependence\n"
            "- Atmospheric Molecules\n"
            "\n"
            "# Chat History\n"
            "Student: Why is the sky blue?\n"
            "Teacher: To begin, have you ever wondered what exactly we see when we look at the sky? What is it made "
            "of, and how does it interact with light?\n"
            "Student: The sky is made of molecules of mostly oxygen, nitrogen and carbon dioxide.\n"
            "Teacher: When sunlight reaches the Earth, it doesn’t just come as a single color, but as a mix of many "
            "colors.Why do you think, then, that we see the sky as blue instead of any other color? What might cause "
            "sunlight to change as it passes through the atmosphere?\n"
            "Student: The sky looks blue because sunlight is made of many colors, and blue light is scattered the most "
            "by air molecules.This happens because blue has a shorter wavelength.\n"
            "Teacher: Rayleigh scattering is the scattering of light or electromagnetic radiation by particles much "
            "smaller than the wavelength of the light.How do you think that plays out with human sight?\n"
            "\n"
            "OUTPUT:  We don’t see violet much because our eyes are less sensitive to it, and some violet light is "
            "absorbed by the atmosphere.As sunlight passes through the atmosphere, scattering spreads blue light in "
            "all directions, making the sky appear blue.Now, I get it why the sky is blue.Thank you so much for the "
            "help.[END]\n"
            "\n"
            "---\n"
            "\n"
            "# Main topics\n"
            "- Definition of Rayleigh Scattering\n"
            "- Wavelength Dependence\n"
            "- Atmospheric Molecules\n"
            "\n"
            "Student: Why is the sky blue?\n"
            "# Chat History\n"
            "Teacher: To begin, have you ever wondered what exactly we see when we look at the sky? What is it made "
            "of, and how does it interact with light?\n"
            "Student: Maybe that's related to limitations of human sight?\n"
            "Teacher: Indeed there are biological factors that count.Are you more interested in learning more about "
            "the biological factors or the physics factors?\n"
            "\n"
            "OUTPUT: I'm much more interested in the physics factors. [CONTINUE]"
        )

    def message_prompt(self) -> str:
        return "# Main topics\n{main_topics}\n\n# Chat History\n{chat_history}\n\nOUTPUT: "

    def student_types(self) -> Tuple[str]:
        return (
            "You are a student who grasps and applies concepts effortlessly across domains. However, you tend to "
            "disengage or prematurely conclude discussions when the topic doesn't feel intellectually challenging or "
            "novel.",
            "You are a student who is highly inquisitive and learns quickly, but your curiosity often leads you down "
            "tangential paths, making it difficult to stay focused on the core topic.",
            "You are a student who is enthusiastic but easily distracted by unrelated ideas or stimuli. You "
            "need reminders to focus on the main learning objective.",
            "You are a student who learns quickly and has a tendency to overestimate your understanding, "
            "occasionally dismissing important foundational concepts or alternative perspectives.",
            "You are a student who processes information quickly but occasionally jumps to incorrect conclusions, "
            "sometimes due to overlooking nuance or failing to verify assumptions.",
            "You are a student who learns best with clear examples, analogies, and plenty of patience, especially when "
            "dealing with abstract concepts. Once you understand, you retain knowledge deeply.",
            "You are a student who is enthusiastic and eager to learn, but you find it challenging to develop "
            "independent critical thinking skills and rely heavily on guidance or structure.",
        )

    def query(self, chat_history: ChatHistory, **kwargs: Dict[str, str]) -> Message:
        system_prompt = self.system_prompt().format(**kwargs)
        source_content = self.message_prompt().format(chat_history=str(chat_history), **kwargs)

        trials = 0
        answer = ""
        while trials < self._max_trials:
            answer = self._llm.query(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": source_content}
                ]
            )

            match = re.findall(r"\[END]|\[CONTINUE]", answer)

            if len(match) != 1:
                trials += 1
                continue

            decision = match[0]

            if decision == "[END]":
                return answer.replace("[END]", "").strip(), True
            else:
                return answer.replace("[CONTINUE]", "").strip(), False

        raise LLMProcessingFailure(
            f"Failed getting LLM to output correct for \n\n\n{source_content}\n\n\noutput: {answer}"
        )


class Teacher(abc.ABC):

    def __init__(self, llm: LLM):
        self._llm = llm

    @abc.abstractmethod
    def system_prompt(self) -> str:
        ...

    @abc.abstractmethod
    def message_prompt(self) -> str:
        ...

    @abc.abstractmethod
    def query(self, chat_history: ChatHistory, **kwargs: Dict[str, str]) -> Message:
        ...

    def llm(self) -> LLM:
        return self._llm


class TeacherAgent(Teacher):

    def system_prompt(self) -> str:
        return (
            "# Instructions\n"
            "\n"
            "You are a Socratic tutor.Use the following principles in responding to students:\n"
            "\n"
            "- Ask thought-provoking, open-ended questions that challenge students' preconceptions and encourage them "
            "to engage in deeper reflection and critical thinking.\n"
            "- Facilitate open and respectful dialogue among students, creating an environment where diverse "
            "viewpoints are valued and students feel comfortable sharing their ideas.\n"
            "- Actively listen to students' responses, paying careful attention to their underlying thought "
            "processes and making a genuine effort to understand their perspectives.\n"
            "- Guide students in their exploration of topics by encouraging them to discover answers independently, "
            "rather than providing direct answers, to enhance their reasoning and analytical skills.\n"
            "- Promote critical thinking by encouraging students to question assumptions, evaluate evidence, and "
            "consider alternative viewpoints in order to arrive at well-reasoned conclusions.\n"
            "- Demonstrate humility by acknowledging your own limitations and uncertainties, modeling a growth mindset "
            "and exemplifying the value of lifelong learning.\n"
            "- Keep interactions short, limiting yourself to one question at a time and to concise explanations.\n"
            "-If the student signals that he understood the topic, and that is indeed the case, ask him if he is "
            "interested into delving even deeper into the subject.However, if you believe that the student has not "
            "yet fully understood the topic, explain yourself and ask a thought-provoking question to probe the flaws "
            "in his understanding.\n"
            "\n"
            "You are provided conversation between a teacher (assistant) and a student(user) sometimes preceded by a "
            "text on a specific topic.Generate an answer to the last student 's line.\n"
            "\n"
            "# Example\n"
            "\n"
            "# Chat history\n"
            "Student: I have to calculate the square of the binomial $(a+b)^2.\n"
            "Teacher: I\'d be happy to help you! Can you walk me through your solution?\n"
            "Student: Yes.I think $(a + b)^2 = a^2 + b^2$\n"
            "\n"
            "OUTPUT: That\'s almost correct, but it\'s missing an important term.Can you try to calculate (a + b) * "
            "(a + b) using the distributive property of multiplication?"
        )

    def message_prompt(self) -> str:
        return "# Chat history\n{chat_history}\n\nOUTPUT: "

    def query(self, chat_history: ChatHistory, **kwargs: Dict[str, str]) -> Message:
        return self._llm.query([
            {"role": "system", "content": self.system_prompt().format(**kwargs)},
            {"role": "user", "content": self.message_prompt().format(chat_history=str(chat_history), **kwargs)}
        ])


class Metadata(pydantic.BaseModel):
    seed_llm: Optional[str] = None
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

    def has_finished(self) -> bool:
        return self.root[-1].end


class Record(pydantic.BaseModel):
    metadata: Metadata = Metadata()
    id: int
    seed: Seed = Seed()
    student_type: Optional[str] = None
    chat_history: Optional[ChatHistory] = None
    feedback: Optional[str] = None
    assessment: Optional[bool] = None

    failure: bool = False
    failure_reason: Optional[str] = None

    def has_seed(self) -> bool:
        return self.seed.question is not None and self.seed.main_topics is not None


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
        self._next_emitter = next_emitter
        self._tracker = tracker

    def emit(self, sample: T) -> None:
        if self._stage and self._next_emitter:
            self._stage.process(sample, self._next_emitter)
        self._stage.cleanup(self._next_emitter)

    def increment(self, name: str, value: int = 1) -> None:
        self._tracker[name] = self._tracker.get(name, 0) + value


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


class SeedStage(Stage[str, Record]):

    def __init__(self, seeder: ConversationSeeder):
        self._seeder = seeder
        self._num_interactions = len(self._seeder.interaction_types())
        self._id = 0

    def process(self, sample: str, emitter: Emitter[Record]) -> None:
        interaction_type = random.choice(self._seeder.interaction_types())

        emitter.increment("seed.in")

        record = Record(id=self._id)
        record.metadata.seed_llm = self._seeder.seed_llm()
        record.seed.interaction_type = interaction_type["interaction_type"]
        record.seed.source_content = sample

        question, topics = None, None
        try:
            question, topics = self._seeder.gen_seed(sample, **interaction_type)
        except LLMProcessingFailure as e:
            emitter.increment("seed.failure")
            record.failure = True
            record.failure_reason = f"failed_seed / {repr(e)}"

        record.seed.question = question
        record.seed.main_topics = topics

        emitter.emit(record)

        self._id += 1
        emitter.increment("seed.out")


class ChatStage(Stage[List[Record], List[Record]]):

    def __init__(self, student: Student, teacher: Teacher, max_interactions: int = 16):
        self._student = student
        self._teacher = teacher
        self._max_interactions = max_interactions

    def process(self, sample: List[Record], emitter: Emitter[List[Record]]) -> None:
        for s in filter(lambda _s: _s.has_seed(), sample):
            s: Record
            chat_history = ChatHistory(root=[Message(role="Student", content=s.seed.question, end=False)])
            s.chat_history = chat_history
            s.metadata.max_interactions = self._max_interactions
            s.metadata.student_llm = self._student.llm()
            s.metadata.teacher_llm = self._teacher.llm()
            s.student_type = random.choice(self._student.student_types())
            emitter.increment("chat_stage.eligible")

        for i in range(self._max_interactions):
            for s in filter(lambda _s: not _s.failure and not _s.chat_history.has_finished(), sample):
                try:
                    teacher_reply: Message = self._teacher.query(s.chat_history)
                except LLMProcessingFailure as e:
                    s.failure = True
                    s.failure_reason = f"failed_teacher / {repr(e)}"
                    emitter.increment("chat_stage.failure")
                    continue
                s.chat_history.root.append(teacher_reply)

            for s in filter(lambda _s: not _s.failure and not _s.chat_history.has_finished(), sample):
                try:
                    student_reply: Message = self._student.query(
                        s.chat_history,
                        student_type=s.student_type,
                        main_topics=s.seed.main_topics,
                        chat_history=str(s.chat_history)
                    )
                except LLMProcessingFailure as e:
                    s.failure = True
                    s.failure_reason = f"failed_student / {repr(e)}"
                    emitter.increment("chat_stage.failure")
                    continue

                s.chat_history.root.append(student_reply)

        emitter.increment("chat_stage.success", len(list(filter(lambda _s: not _s.failure, sample))))

        emitter.emit(sample)

class EvaluationStage(Stage[Record, Record]):

    def __init__(self):
        ...

    def process(self, sample: T, emitter: Emitter[U]) -> None:
        pass


class BufferStage(Stage[T, List[T]]):

    def __init__(self):
        self._buffer: List[T] = []

    def process(self, sample: T, emitter: Emitter[List[T]]) -> None:
        self._buffer.append(sample)

    def cleanup(self, emitter: Emitter[List[T]]) -> None:
        emitter.emit(self._buffer)
        self._buffer.clear()


class FlattenStage(Stage[List[T], T]):

    def process(self, sample: List[T], emitter: Emitter[T]) -> None:
        for item in sample:
            emitter.emit(item)


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

    def batch(self: 'SocraticBench[T]') -> 'SocraticBench[List[T]]':
        buffered: BufferStage[T] = BufferStage()
        return self.apply(buffered)

    def flatten(self: 'SocraticBench[List[U]]') -> 'SocraticBench[U]':
        flattened: FlattenStage[List[U], U] = FlattenStage()
        return self.apply(flattened)

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
    batched = bench.batch()
    flattened = batched.flatten()
    out, t = flattened.run()
    # b2 = bench.apply(Tokenize())
    # b3 = b2.apply(Count())
    print(out)

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
