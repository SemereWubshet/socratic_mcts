import argparse
import pathlib
import random
import shutil
from typing import Optional, List, Tuple, Dict, Union

from datasets import DatasetDict, load_dataset
from ollama import Client
from openai import OpenAI
from pydantic import BaseModel, RootModel
from tqdm import tqdm

from agents import StudentSeed, LLM, Student, Teacher, Judge, OpenAIAgent, OllamaAgent, Socratic


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


def resolve_llm(model: Tuple[str, str], clients: Dict[str, Union[Client, OpenAI]]) -> LLM:
    backend, model = model

    client = clients.get(backend)

    if not client:
        raise ValueError(f"No client configured for backend '{backend}'")

    llm: LLM
    if backend == "openai":
        llm = OpenAIAgent(model, client, temperature=0.15)
    elif backend == "ollama":
        llm = OllamaAgent(model, client, temperature=0.15, num_ctx=8192)
    else:
        raise ValueError(f"Backend {backend} is not supported")

    llm.healthcheck()

    return llm


def gen_seeds(
        wikipedia_pages: DatasetDict,
        llm: LLM,
        num_of_conversations: int,
        max_page_size: int = 3000
) -> SeedDataset:
    contents = wikipedia_pages['train']
    contents = contents.shuffle()

    pbar = tqdm(total=num_of_conversations, desc="Conversation seeds")
    seeds = []
    i = 0
    while len(seeds) < num_of_conversations and i < len(contents):
        page = contents[i]
        page_text: str = page["chapter"][:max_page_size]
        i += 1

        seed_type = random.randint(0, len(StudentSeed.INTERACTION_TYPES)) - 1
        seeder = StudentSeed(llm, seed_type)
        question, main_topics = seeder.gen_seed(page_text)
        seeds.append(
            Seed(source_content=page_text,
                 question=question,
                 main_topics=main_topics,
                 interaction_type=StudentSeed.INTERACTION_TYPES[seed_type]["interaction_type"])
        )
        pbar.update()

    return SeedDataset(seeds)


def gen_teacher_student_interactions(
        seeds: SeedDataset,
        student_llm: LLM,
        teacher: Teacher,
        max_interactions: int = 3
) -> InteractionDataset:
    interactions: List[Interaction] = []
    students: List[Student] = []
    ended: List[bool] = []

    total_possible_queries = 2 * len(seeds.root) * max_interactions
    pbar = tqdm(total=total_possible_queries, desc="Simulating dialogs", unit="query")

    # Prepare initial state
    for seed in seeds.root:
        stype = random.choice(Student.TYPES)
        student = Student(student_llm, seed.main_topics, stype)

        history = ChatHistory(root=[Message(role="Student", content=seed.question, end=False)])
        interactions.append(Interaction(seed=seed, student_type=stype, chat_history=history))
        students.append(student)
        ended.append(False)

    # Interleave teacher/student in rounds
    for turn in range(max_interactions):
        # -- TEACHER step --
        teacher_prompts = []
        idx_map = []
        for idx, inter in enumerate(interactions):
            if not ended[idx]:
                teacher_prompts.append(str(inter.chat_history))
                idx_map.append(idx)

        if not teacher_prompts:
            break  # All done

        teacher_replies = [teacher.chat(p) for p in teacher_prompts]
        pbar.update(len(teacher_replies))

        for batch_i, reply in enumerate(teacher_replies):
            idx = idx_map[batch_i]
            interactions[idx].chat_history.root.append(
                Message(role="Teacher", content=reply, end=False)
            )

        # -- STUDENT step --
        student_prompts = []
        idx_map = []
        for idx, inter in enumerate(interactions):
            if not ended[idx]:
                student_prompts.append(str(inter.chat_history))
                idx_map.append(idx)

        student_outputs = [students[idx].chat(p) for idx, p in zip(idx_map, student_prompts)]
        pbar.update(len(student_outputs))

        for (reply, did_end), idx in zip(student_outputs, idx_map):
            interactions[idx].chat_history.root.append(
                Message(role="Student", content=reply, end=did_end)
            )

            if did_end:
                ended[idx] = True
                # Pre-advance the bar for skipped steps: 2 (T+S) per remaining turn
                remaining_turns = max_interactions - (turn + 1)
                skipped_steps = 2 * remaining_turns
                pbar.update(skipped_steps)

        if all(ended):
            break

    pbar.close()
    return InteractionDataset(
        metadata=InteractionMetadata(
            student_llm=student_llm.model_name,
            teacher_llm=teacher.model_name(),
            max_interactions=max_interactions
        ),
        interactions=interactions
    )


def evaluate(interactions: InteractionDataset, judge_llm: LLM) -> EvaluationDataset:
    judge = Judge(judge_llm)

    evaluations = []
    for _id, interaction in enumerate(tqdm(interactions.interactions, desc="Evaluating interactions")):
        feedback, assessment = judge.evaluate(interaction.seed.main_topics, str(interaction.chat_history))
        evaluations.append(Evaluation(id=_id, interaction=interaction, feedback=feedback, assessment=assessment))

    return EvaluationDataset(
        metadata=EvalMetadata(
            student_llm=interactions.metadata.student_llm,
            teacher_llm=interactions.metadata.teacher_llm,
            judge_llm=judge_llm.model_name,
            max_interactions=max_interactions
        ),
        evaluations=evaluations
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ROLLOUT", description="Rollouts conversations between student and teacher and assess them."
    )

    parser.add_argument("--output-dir", required=True, type=str, help="Path where to store pipeline outputs")
    parser.add_argument("--num-conversations", required=True, type=int, help="Number of conversations to generate")
    parser.add_argument("--ollama-client", required=True, type=str, help="The address for ollama server.")

    parser.add_argument(
        "--seed-llm", nargs=2, metavar=("BACKEND", "MODEL"), default=["ollama", "mistral-small3.1:24b"]
    )
    parser.add_argument(
        "--student-llm", nargs=2, metavar=("BACKEND", "MODEL"), default=["ollama", "mistral-small3.1:24b"]
    )
    parser.add_argument(
        "--judge-llm", nargs=2, metavar=("BACKEND", "MODEL"), default=["ollama", "qwen3:32b"]
    )

    parser.add_argument("--use-cache", action="store_true", help="Don't run subprocess if output files exist")
    args = parser.parse_args()

    clients = {"openai": OpenAI(), "ollama": Client(args.ollama_client)}

    seed_llm = resolve_llm(args.seed_llm, clients)
    student_llm = resolve_llm(args.student_llm, clients)
    judge_llm = resolve_llm(args.judge_llm, clients)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    if not args.use_cache:
        for child in output_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)

    seeds_path = output_dir / "seeds.json"

    wikipedia = load_dataset("princeton-nlp/TextbookChapters")

    if seeds_path.exists():
        print("Loading seed dataset", flush=True)
        seed_dataset = SeedDataset.model_validate_json(seeds_path.read_text())
    else:
        seed_dataset = gen_seeds(wikipedia, seed_llm, num_of_conversations=args.num_conversations)
        seeds_path.write_text(seed_dataset.model_dump_json(indent=4))
        # Remove stale downstream outputs if seeds changed
        for f in output_dir.glob("int_*.json"):
            f.unlink()
        for f in output_dir.glob("eval_*.json"):
            f.unlink()

    for teacher in tqdm([
        Teacher(resolve_llm(("ollama", "mistral-small3.1:24b"), clients)),
        # Teacher(resolve_llm(("ollama", "llama3.3:70b"), clients)),
        # Teacher(resolve_llm(("openai", "gpt-4o"), clients)),
        Socratic(resolve_llm(("ollama", "eurecom-ds/phi-3-mini-4k-socratic"), clients)),
    ], desc="Teacher evaluation"):
        for max_interactions in tqdm([2, 4, 8, 16], desc="Max interactions"):
            teacher_model = teacher.model_name()
            teacher_model = teacher_model.split("/")[-1].replace(":", "_")
            interactions_path = output_dir / f"int_{max_interactions}_{teacher_model}.json"
            evaluations_path = output_dir / f"eval_{max_interactions}_{teacher_model}.json"

            if interactions_path.exists():
                interactions_dataset = InteractionDataset.model_validate_json(interactions_path.read_text())
            else:
                interactions_dataset = gen_teacher_student_interactions(
                    seed_dataset, student_llm, teacher, max_interactions=max_interactions
                )
                interactions_path.write_text(interactions_dataset.model_dump_json(indent=4))
                evaluations_path.unlink(missing_ok=True)

            if not evaluations_path.exists():
                evaluations_dataset = evaluate(interactions_dataset, judge_llm)
                evaluations_path.write_text(evaluations_dataset.model_dump_json(indent=4))

    print("Finished processing")
    exit(0)
