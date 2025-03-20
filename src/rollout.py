import argparse
import pathlib
import random
import shutil
from typing import Optional

import numpy as np
import ollama
from datasets import DatasetDict, load_dataset
from pydantic import BaseModel, RootModel
from tqdm import tqdm

from agents import StudentSeed, LLM, Student, Teacher, Judge, OllamaAgent
from base_models import ChatHistory, Message
from tools import LLMAction


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


class Interaction(BaseModel):
    seed: Seed
    student_type: str
    chat_history: ChatHistory

    def __str__(self) -> str:
        return f"{self.seed}\n\n# Student type\n{self.student_type}\n\n# Chat History\n{self.chat_history}"


class InteractionDataset(RootModel):
    root: list[Interaction]

    def __str__(self) -> str:
        return "\n\n-------\n\n".join(str(i) for i in self.root)


class Evaluation(BaseModel):
    id: int
    interaction: Interaction
    feedback: Optional[str]
    assessment: Optional[bool]

    def __str__(self) -> str:
        return f"{self.interaction}\n\n# Feedback\n{self.feedback}\n\n# Assessment\n{self.assessment}"


class EvaluationDataset(RootModel):
    root: list[Evaluation]

    def __str__(self) -> str:
        return "\n\n-------\n\n".join(str(e) for e in self.root)

    def avg_performance(self) -> float:
        return np.mean([e.assessment for e in self.root])


def gen_seeds(
        wikipedia_pages: DatasetDict,
        llm: LLM,
        num_of_conversations: int,
        max_chunk_size: int = 2000
) -> SeedDataset:
    contents = wikipedia_pages['train']

    seeds = []
    for _ in tqdm(range(num_of_conversations)):
        page = random.choice(contents)
        page_text: str = page["text"]
        start_index = random.randint(0, max(0, len(page_text) - max_chunk_size))
        random_chunk = page_text[start_index:start_index + max_chunk_size]
        seed_type = random.randint(0, len(StudentSeed.INTERACTION_TYPES)) - 1
        seeder = StudentSeed(llm, seed_type)
        question, main_topics = seeder.gen_seed(random_chunk)
        seeds.append(
            Seed(source_content=page_text,
                 question=question,
                 main_topics=main_topics,
                 interaction_type=StudentSeed.INTERACTION_TYPES[seed_type]["interaction_type"])
        )

    return SeedDataset(seeds)


def gen_teacher_student_interactions(
        seeds: SeedDataset,
        student_llm: LLM,
        teacher: Teacher,
        max_interactions: int = 3
) -> InteractionDataset:
    interactions = InteractionDataset([])
    for seed in tqdm(seeds.root):
        type_idx = random.randint(0, len(Student.TYPES) - 1)
        student = Student(student_llm, seed.main_topics, Student.TYPES[type_idx])

        chat_history = ChatHistory([])
        chat_history.root.append(Message(role="Student", content=seed.question, end=False))

        for _ in range(max_interactions):
            reply = teacher.chat(chat_history)
            chat_history.root.append(Message(role="Teacher", content=reply, end=False))
            reply, end = student.chat(chat_history)
            chat_history.root.append(Message(role="Student", content=reply, end=end))
            if end:
                break

        interactions.root.append(
            Interaction(seed=seed, student_type=student.TYPES[type_idx], chat_history=chat_history)
        )

    return interactions


def evaluate(interactions: InteractionDataset, judge_llm: LLM) -> EvaluationDataset:
    judge = Judge(judge_llm)

    evaluations = EvaluationDataset([])
    for _id, interaction in enumerate(tqdm(interactions.root)):
        feedback, assessment = judge.evaluate(interaction.seed.main_topics, str(interaction.chat_history))
        evaluations.root.append(Evaluation(id=_id, interaction=interaction, feedback=feedback, assessment=assessment))

    return evaluations


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="ROLLOUT", description="Rollouts conversations between student and teacher and assess them."
    )

    parser.add_argument("--output-dir", required=True, type=str, help="Path where to store pipeline outputs")
    parser.add_argument("--num-conversations", required=True, type=int, help="Number of conversations to generate")
    parser.add_argument("--max-interactions", default=15, type=int,
                        help="Maximum number of conversations rounds between the teacher and the student")

    parser.add_argument(
        "--seed-llm", nargs=3, action=LLMAction,
        help="Service to create the seed topics. It can be either a self-hosted model (Ollama) or OpenAI."
             " This argument expects 3 parameters. The service to use: openai or ollama. The access "
             "information: if openai, thus the OpenAi API key or if using ollama, the server's http "
             "address. The last parameter is the model to use (e.g., gpt-4o or llama3:70b-instruct).",
        default=OllamaAgent("mistral-nemo:12b-instruct-2407-fp16", ollama.Client("http://atlas1api.eurecom.fr"))
    )
    parser.add_argument(
        "--student-llm", nargs=3, action=LLMAction,
        help="Service to emulate the student. It can be either a self-hosted model (Ollama) or OpenAI."
             " This argument expects 3 parameters. The service to use: openai or ollama. The access "
             "information: if openai, thus the OpenAi API key or if using ollama, the server's http "
             "address. The last parameter is the model to use (e.g., gpt-4o or llama3:70b-instruct).",
        default=OllamaAgent("mistral-nemo:12b-instruct-2407-fp16", ollama.Client("http://atlas1api.eurecom.fr"))
    )
    parser.add_argument(
        "--teacher-llm", nargs=3, action=LLMAction,
        help="Service to emulate the teacher. It can be either a self-hosted model (Ollama) or OpenAI."
             " This argument expects 3 parameters. The service to use: openai or ollama. The access "
             "information: if openai, thus the OpenAi API key or if using ollama, the server's http "
             "address. The last parameter is the model to use (e.g., gpt-4o or llama3:70b-instruct).",
        default=OllamaAgent("mistral-nemo:12b-instruct-2407-fp16", ollama.Client("http://atlas1api.eurecom.fr"))
    )
    parser.add_argument(
        "--judge-llm", nargs=3, action=LLMAction,
        help="Service to use of the judge LLM. It can be either a self-hosted model (Ollama) or OpenAI."
             " This argument expects 3 parameters. The service to use: openai or ollama. The access "
             "information: if openai, thus the OpenAi API key or if using ollama, the server's http "
             "address. The last parameter is the model to use (e.g., gpt-4o or llama3:70b-instruct).",
        default=OllamaAgent("llama3.3:70b", ollama.Client("http://atlas1api.eurecom.fr"))
    )

    parser.add_argument("--use-cache", action="store_true", help="Don't run subprocess if output files exist")
    parser.add_argument("--human-eval", action="store_true", help="Produce a masked version of the evaluation dataset")
    args = parser.parse_args()

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    if not args.use_cache:
        for child in output_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)

    seeds_path = output_dir / "seeds.json"
    interactions_path = output_dir / "interactions.json"
    evaluations_path = output_dir / "evaluations.json"
    human_eval_path = output_dir / "human-eval.json"

    wikipedia = load_dataset("wikimedia/wikipedia", "20231101.simple")

    if seeds_path.exists():
        print("Loading seed dataset", flush=True)
        seed_dataset = SeedDataset.model_validate_json(seeds_path.read_text())
    else:
        print("Creating seed dataset", flush=True)
        seed_dataset = gen_seeds(wikipedia, args.seed_llm, num_of_conversations=args.num_conversations)
        seeds_path.write_text(seed_dataset.model_dump_json(indent=4))
        interactions_path.unlink(missing_ok=True)
        evaluations_path.unlink(missing_ok=True)

    if interactions_path.exists():
        print("Loading interactions dataset", flush=True)
        interactions_dataset = InteractionDataset.model_validate_json(interactions_path.read_text())
    else:
        print()
        print("Creating interactions dataset", flush=True)
        interactions_dataset = gen_teacher_student_interactions(
            seed_dataset, args.student_llm, Teacher(args.teacher_llm), max_interactions=args.max_interactions
        )
        interactions_path.write_text(interactions_dataset.model_dump_json(indent=4))
        evaluations_path.unlink(missing_ok=True)

    if evaluations_path.exists() and not (args.human_eval and not human_eval_path.exists()):
        print("Are you sure about what you are doing? Evaluation dataset exists already, but "
              "nothing else was regenerated.")
        exit(1)

    if evaluations_path.exists():
        print("Loading evaluation dataset", flush=True)
        evaluations_dataset = EvaluationDataset.model_validate_json(evaluations_path.read_text())
    else:
        print()
        print("Assessing teacher-student interactions", flush=True)
        evaluations_dataset = evaluate(interactions_dataset, args.judge_llm)
        evaluations_path.write_text(evaluations_dataset.model_dump_json(indent=4))
        human_eval_path.unlink(missing_ok=True)

    if args.human_eval:
        for evaluation in evaluations_dataset.root:
            evaluation.assessment = None
            evaluation.feedback = None
        human_eval_path.write_text(evaluations_dataset.model_dump_json(indent=4))

    print("Finished processing")
    exit(0)
