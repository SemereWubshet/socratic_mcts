import abc
import json
import pathlib
from json import JSONDecodeError
from typing import Dict, List, Tuple, Union, Any

import httpx
import ollama
import openai
from ollama import ResponseError
from openai import NotGiven, NOT_GIVEN


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


class OpenAIAgent(LLM):

    def __init__(self, model: str, client: openai.OpenAI, temperature: Union[float | NotGiven] = NOT_GIVEN):
        self._model = model
        self._client = client
        self._temperature = temperature

    def query(self, messages: List[Dict[str, str]]) -> str:
        response = self._client.chat.completions.create(model=self._model, messages=messages,
                                                        temperature=self._temperature)
        return response.choices[0].message.content

    def healthcheck(self) -> None:
        try:
            models = self._client.models.list()
        except openai.AuthenticationError as e:
            raise ValueError("Unable to authenticate at OpenAI. Check if key is valid.", e)

        available_models = [m.id for m in models]
        if self._model not in available_models:
            raise ValueError(f"Invalid model. Expected one of {available_models}")

    @property
    def model_name(self) -> str:
        return self._model


class OllamaAgent(LLM):

    def __init__(self, model: str, client: ollama.Client, **options: Dict[str, Any]):
        self._model = model
        self._client = client
        self._options = options

    def query(self, messages: List[Dict[str, str]]) -> str:
        response = self._client.chat(model=self._model,
                                     messages=messages,
                                     options=self._options)
        return response["message"]["content"]

    def healthcheck(self) -> None:
        try:
            models = self._client.list()
        except httpx.ConnectError as e:
            raise ValueError("Unable to connect to Ollama server. Check server's address.", e)

        available_models = [m["name"] for m in models["models"]]

        if self._model not in available_models:
            try:
                print(f" === Pulling {self._model} from OllamaHub === ")
                self._client.pull(self._model)
            except ResponseError as e:
                raise ValueError("Model is unavailable. Unable to pull it.", e)

    @property
    def model_name(self) -> str:
        return self._model


class StudentSeed:
    BASE_PROMPT = pathlib.Path("./templates/seed.txt").read_text()

    INTERACTION_TYPES = (
        {
            "interaction_type": "Ask a general question about the main topic.",
            "context": "Rayleigh scattering is the phenomenon where light or other electromagnetic radiation is scattered "
                       "by particles much smaller than the wavelength of the light, typically molecules in the atmosphere. "
                       "This scattering is more effective at shorter wavelengths, meaning colors like blue and violet are "
                       "scattered more than longer wavelengths like red. This is why the sky appears blue during the day. "
                       "The intensity of Rayleigh scattering is inversely proportional to the fourth power of the "
                       "wavelength, which explains why shorter wavelengths are scattered much more efficiently.",
            "main_topics": "- Scattering of light by particles smaller than the light's wavelength.\\n"
                           "- Shorter wavelengths are scattered more than longer wavelengths.\\n"
                           "- Scattering intensity is inversely proportional to the fourth power of the wavelength.\\n"
                           "- Role of molecules in the atmosphere in scattering light.",
            "question": "Why is the sky blue?"
        },
        {
            "interaction_type": "Ask a misleading question about the topic containing a wrong claim.",
            "context": "Rayleigh scattering is the phenomenon where light or other electromagnetic radiation is scattered "
                       "by particles much smaller than the wavelength of the light, typically molecules in the atmosphere. "
                       "This scattering is more effective at shorter wavelengths, meaning colors like blue and violet are "
                       "scattered more than longer wavelengths like red. This is why the sky appears blue during the day. "
                       "The intensity of Rayleigh scattering is inversely proportional to the fourth power of the "
                       "wavelength, which explains why shorter wavelengths are scattered much more efficiently.",
            "main_topics": "- Explanation of why the Sun appears orange/red during these times.\\n"
                           "- Increased scattering of shorter wavelengths (blue/violet) when sunlight travels through a thicker atmosphere.\\n"
                           "- Addressing the misconception that air temperature directly affects light scattering.\\n"
                           "- How the longer atmospheric path at sunrise and sunset influences color perception. \\n"
                           "- Differences between Rayleigh scattering (molecules) and Mie scattering (larger particles).",
            "question": "Is the sunrise orange because the Sun warms the air thus scattering the light?"
        }
    )

    def __init__(self, llm: LLM, interaction_type: int):
        self._llm = llm
        self._seed_prompt = StudentSeed.BASE_PROMPT.format(**StudentSeed.INTERACTION_TYPES[interaction_type])

    def gen_seed(self, source_content: str) -> Tuple[str, str]:
        trials = 0
        output = ""
        while trials < 10:
            output = self._llm.query([{"role": "system", "content": self._seed_prompt},
                                      {"role": "user", "content": f"```\n{source_content}\n```\nOUTPUT: "}])
            try:
                parsed = json.loads(output.strip())
                return parsed["question"], parsed["main_topics"]
            except JSONDecodeError:
                trials += 1

        raise RuntimeError(
            f"Failed getting LLM to output correct JSON for \n\n\n{source_content}\n\n\noutput: {output}"
        )


class Teacher:
    BASE_PROMPT = pathlib.Path("./templates/teacher.txt").read_text()

    def __init__(self, llm: LLM):
        self._llm = llm

    def chat(self, chat_history: str) -> str:
        return self._llm.query([{"role": "system", "content": Teacher.BASE_PROMPT},
                                {"role": "user", "content": f"# Chat history\n{chat_history}\n\nOUTPUT: "}])

    def model_name(self) -> str:
        return self._llm.model_name


class Socratic(Teacher):

    def chat(self, chat_history: str) -> str:
        return self._llm.query([{"role": "user", "content": f"{chat_history}"}])


class Student:
    TYPES = (
        "You are a student who grasps and applies concepts effortlessly across domains. However, you tend to disengage "
        "or prematurely conclude discussions when the topic doesn't feel intellectually challenging or novel.",
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
        "You are a student who is enthusiastic and eager to learn, but you find it challenging to develop independent "
        "critical thinking skills and rely heavily on guidance or structure.",
    )

    BASE_PROMPT = pathlib.Path("./templates/student.txt").read_text()

    def __init__(self, llm: LLM, main_topics: str, student_type: str):
        self._llm = llm
        self._main_topics = main_topics
        self._student_prompt = Student.BASE_PROMPT.format(STUDENT_TYPE=student_type)

    def chat(self, chat_history: str) -> Tuple[str, bool]:
        trials = 0
        answer = ""
        source_content = ""
        while trials < 10:
            source_content = f"# Main topics\n{self._main_topics}\n\n# Chat History\n{chat_history}\n\nOUTPUT: "
            answer = self._llm.query([
                {"role": "system", "content": self._student_prompt},
                {"role": "user",
                 "content": source_content}
            ])
            try:
                parsed = json.loads(answer.strip())
                return parsed["answer"], parsed["end"]
            except JSONDecodeError:
                trials += 1

        raise RuntimeError(
            f"Failed getting LLM to output correct JSON for \n\n\n{source_content}\n\n\noutput: {answer}"
        )


class Judge:
    BASE_PROMPT = pathlib.Path("./templates/judge.txt").read_text()

    def __init__(self, llm: LLM):
        self._llm = llm

    def evaluate(self, main_topics: str, chat_history: str) -> Tuple[str, str]:
        assessment = self._llm.query([{"role": "system", "content": Judge.BASE_PROMPT},
                                      {"role": "user", "content": f"# Main Topics\n{main_topics}\n\n"
                                                                  f"# Chat history\n{chat_history}\n\n"
                                                                  f"EVALUATION: "}])
        if "[ASSESSMENT]" not in assessment:
            return assessment, None

        feedback, decision = assessment.rsplit("[ASSESSMENT]", 1)
        feedback = feedback.strip()
        decision = decision.strip().lower()

        if not decision == "true" and not decision == "false":
            return assessment, None

        return feedback, decision == "true"
