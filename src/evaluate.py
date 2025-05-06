import argparse
import pathlib
import shutil
from typing import Optional, Tuple

import numpy as np
import ollama
from datasets import load_dataset
from openai import OpenAI
from pydantic import BaseModel

from agents import OllamaAgent, Teacher, OpenAIAgent, LLM, Student, Judge
from mcts import StudentNode, ValueFn, select, backup, TeacherNode
from rollout import SeedDataset, gen_seeds, gen_teacher_student_interactions, Evaluation, evaluate
from tools import LLMAction


def expand(node: TeacherNode,
           student: Student,
           teacher: Teacher,
           value_fn: ValueFn,
           judge: Judge,
           main_topics: str,
           max_depth: int = 15) -> Tuple[bool, StudentNode]:
    # Adapted from mcts.py
    student_node = node.expand(student)

    if student_node.end or student_node.depth() >= max_depth:
        chat_history = str(student_node.history())
        _, assessment = judge.evaluate(main_topics, chat_history)
        student_node.v = 1. if assessment else -1.
        student_node.terminal = True
        return True, student_node

    student_node.v = value_fn(str(student_node.history()))

    student_node.terminal = False
    student_node.expand(teacher)
    student_node.expand(teacher)
    student_node.expand(teacher)

    return False, student_node


class ResultDataset(BaseModel):
    model_name: str
    # mcts_budget: Optional[int] = None
    evaluations: list[Evaluation]

    def avg_performance(self) -> float:
        return np.mean([e.assessment for e in self.evaluations])


class Socratic(Teacher):

    def chat(self, chat_history: str) -> str:
        return self._llm.query([{"role": "user", "content": f"{chat_history}"}])


class MCTS(Teacher):

    def __init__(self, llm: LLM, judge: Judge, value_fn: ValueFn, budget: int, max_depth: int = 15):
        super().__init__(llm)
        self._value_fn = value_fn
        self._budget = budget
        self._max_depth = max_depth
        self._judge = judge

    def chat(self, chat_history: str) -> str:
        # get main_topics
        main_topics = self._llm.query(
            [{"role": "system", "content": "Given a conversation between a teacher and a "
                                           "student, output a short description of the main topics (up to "
                                           "three) the teacher must cover so to improve the understanding "
                                           "of the student on the topic."},
             {"role": "user", "content": chat_history}])

        # get likely student type
        student_type_list = '\n - '.join(Student.TYPES)
        student_type = self._llm.query(
            [{"role": "system", "content": "Given a conversation between a teacher and a "
                                           "student, output a short description the most likely type of student "
                                           "the teacher is interacting with. Select one of the items from the "
                                           "following list: "
                                           "\n"
                                           f"{student_type_list}"
                                           "\n\nOutput, only the selected choice with the exact text, but no opening "
                                           "or closing explanations."},
             {"role": "user", "content": chat_history}])
        student = Student(self._llm, main_topics, student_type)

        root = StudentNode(chat_history)
        root.v = self._value_fn(str(root.history()))
        flat_teacher = Teacher(self._llm)
        root.expand(flat_teacher)
        root.expand(flat_teacher)
        root.expand(flat_teacher)

        for _ in range(self._budget):
            selected = select(root)
            _, leaf = expand(selected, student, flat_teacher, self._value_fn, self._judge, main_topics, self._max_depth)
            backup(leaf)

        q = np.array([c.q for c in root.children])
        idx = int(np.argmax(q))
        selected = root.children[idx]

        return selected.reply


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="EVALUATE", description="Tool for evaluating and comparing multiple Teacher LLMs."
    )

    parser.add_argument("--root-dir", required=True, type=pathlib.Path, help="Path to root training directory")
    parser.add_argument("--value-fn", required=True, type=pathlib.Path, help="Path to fine-tuned value function")
    parser.add_argument("--num-conversations", required=True, type=int, help="Number of conversations to generate")
    parser.add_argument("--max-interactions", default=15, type=int,
                        help="Maximum number of conversations rounds between the teacher and the student")

    parser.add_argument(
        "--seed-llm", nargs=3, action=LLMAction,
        help="Service to create the seed topics. It can be either a self-hosted model (Ollama) or OpenAI."
             " This argument expects 3 parameters. The service to use: openai or ollama. The access "
             "information: if openai, thus the OpenAi API key or if using ollama, the server's http "
             "address. The last parameter is the model to use (e.g., gpt-4o or llama3:70b-instruct).",
        default=OllamaAgent(
            "mistral-nemo:12b",
            ollama.Client("http://atlas1api.eurecom.fr"),
            temperature=0.,
            num_ctx=32_000
        )
    )
    parser.add_argument(
        "--student-llm", nargs=3, action=LLMAction,
        help="Service to emulate the student. It can be either a self-hosted model (Ollama) or OpenAI."
             " This argument expects 3 parameters. The service to use: openai or ollama. The access "
             "information: if openai, thus the OpenAi API key or if using ollama, the server's http "
             "address. The last parameter is the model to use (e.g., gpt-4o or llama3:70b-instruct).",
        default=OllamaAgent(
            "mistral-nemo:12b",
            ollama.Client("http://atlas1api.eurecom.fr"),
            temperature=0.,
            num_ctx=32_000
        )
    )
    parser.add_argument(
        "--ollama-client", type=str, help="Ollama client to be used by Open Source teacher models.",
        default="http://atlas1api.eurecom.fr"
    )
    parser.add_argument(
        "--judge-llm", nargs=3, action=LLMAction,
        help="Service to use of the judge LLM. It can be either a self-hosted model (Ollama) or OpenAI."
             " This argument expects 3 parameters. The service to use: openai or ollama. The access "
             "information: if openai, thus the OpenAi API key or if using ollama, the server's http "
             "address. The last parameter is the model to use (e.g., gpt-4o or llama3:70b-instruct).",
        default=OllamaAgent(
            "llama3.3:70b",
            ollama.Client("http://atlas1api.eurecom.fr"), temperature=0., num_ctx=32_000
        )
    )

    parser.add_argument("--use-cache", action="store_true", help="Don't run subprocess if output files exist")
    args = parser.parse_args()

    output_dir: pathlib.Path = args.root_dir / "evaluation"

    output_dir.mkdir(exist_ok=True)
    if not args.use_cache:
        for child in output_dir.iterdir():
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                shutil.rmtree(child)

    seeds_path = output_dir / "seeds.json"
    results_path = output_dir / "results"

    results_path.mkdir(exist_ok=True)

    if seeds_path.exists():
        print("Loading seed dataset", flush=True)
        seed_dataset = SeedDataset.model_validate_json(seeds_path.read_text())
    else:
        print("Creating seed dataset", flush=True)
        wikipedia = load_dataset("wikimedia/wikipedia", "20231101.simple")
        seed_dataset = gen_seeds(wikipedia, args.seed_llm, num_of_conversations=args.num_conversations)
        seeds_path.write_text(seed_dataset.model_dump_json(indent=4))

    nemo = OllamaAgent(
        "mistral-nemo:12b", ollama.Client(args.ollama_client), temperature=0., num_ctx=32_000
    )
    nemo.healthcheck()

    socratic_llm = OllamaAgent("eurecom-ds/phi-3-mini-4k-socratic", ollama.Client(args.ollama_client), temperature=0.)
    socratic_llm.healthcheck()

    gpt4o = OpenAIAgent("gpt-4o", OpenAI(), temperature=0.)
    gpt4o.healthcheck()

    nemo_mcts = OllamaAgent(
        "mistral-nemo:12b", ollama.Client(args.ollama_client), temperature=1.7, num_ctx=32_000
    )
    nemo_mcts.healthcheck()

    llama3 = OllamaAgent(
        "llama3.3:70b", ollama.Client(args.ollama_client), temperature=0., num_ctx=32_000
    )
    llama3.healthcheck()

    deepseek = OllamaAgent(
        "deepseek-1:8b", ollama.Client(args.ollama_client), temperature=0., num_ctx=32_000
    )
    deepseek.healthcheck()

    judge = Judge(args.judge_llm)

    value_fn = ValueFn(base_model=str(args.value_fn), gpu="cpu")

    for filename, teacher in ([
        ("mistral-nemo.json", Teacher(nemo)),
        ("deepseek-1.json", Teacher(deepseek)),
        ("socratic-llm.json", Socratic(socratic_llm)),
        ("gpt-4o.json", Teacher(gpt4o)),
        ("llama3.3.json", Teacher(llama3)),
        (f"mcts-budget-0.json", MCTS(nemo_mcts, judge, value_fn, budget=0)),
        (f"mcts-budget-2.json", MCTS(nemo_mcts, judge, value_fn, budget=2)),
        (f"mcts-budget-4.json", MCTS(nemo_mcts, judge, value_fn, budget=4)),
        (f"mcts-budget-8.json", MCTS(nemo_mcts, judge, value_fn, budget=8)),
        (f"mcts-budget-16.json", MCTS(nemo_mcts, judge, value_fn, budget=16)),
    ]):
        print()
        print(f"Evaluating for {filename}...")
        result_path = results_path / filename
        if not result_path.exists():
            interactions_dataset = gen_teacher_student_interactions(
                seed_dataset, args.student_llm, teacher, max_interactions=args.max_interactions
            )
            evaluations_dataset = evaluate(interactions_dataset, args.judge_llm)

            result_dataset = ResultDataset(
                model_name=teacher.model_name(), mcts_budget=None, evaluations=evaluations_dataset.root
            )

            result_path.write_text(result_dataset.model_dump_json(indent=4))
            print(f"Final avg. performance {result_dataset.avg_performance()}")
