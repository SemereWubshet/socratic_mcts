import argparse
import gc
import os
import pathlib
import random
import shutil
from typing import List, Tuple, Dict, Union

import torch
import unsloth
from datasets import DatasetDict, load_dataset
from ollama import Client
from openai import OpenAI
from tqdm import tqdm

from agents import StudentSeed, LLM, Student, Teacher, Judge, OpenAIAgent, OllamaAgent
from schemas import SeedDataset, Seed, InteractionDataset, Interaction, ChatHistory, Message, InteractionMetadata, \
    EvaluationDataset, Evaluation, EvalMetadata


class Qwen(LLM):

    def __init__(self, base_model: str, max_length: int = 1024):
        self._base_model = base_model
        self.max_length = max_length
        self.model = None
        self.tokenizer = None

    def query(self, messages: List[Dict[str, str]], temperature: float = 0.9) -> str:
        if getattr(self, "model", None) is None or self.tokenizer is None:
            self.load()

        raw_prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = self.tokenizer([raw_prompt], return_tensors="pt").to("cuda")
        outputs = self.model.generate(
            **inputs, max_new_tokens=128, do_sample=True, temperature=temperature
        )
        generation = outputs[0, len(inputs['input_ids'][0]):]
        decoded = self.tokenizer.decode(generation, skip_special_tokens=True)
        return decoded

    def load(self, for_inference: bool = True) -> None:
        self.model, self.tokenizer = unsloth.FastLanguageModel.from_pretrained(
            model_name=self._base_model,
            dtype=torch.bfloat16,
            max_seq_length=self.max_length,
            load_in_4bit=False,  # False for LoRA 16bit
            load_in_8bit=False,
        )

        if for_inference:
            self.model = unsloth.FastLanguageModel.for_inference(self.model)

        # ✅ Patch apply_chat_template to default enable_thinking=False
        if hasattr(self.tokenizer, "apply_chat_template"):
            original_fn = self.tokenizer.apply_chat_template

            def patched_apply_chat_template(conversation, **kwargs):
                kwargs.setdefault("enable_thinking", False)
                return original_fn(conversation, **kwargs)

            self.tokenizer.apply_chat_template = patched_apply_chat_template

    def healthcheck(self) -> None:
        pass

    @property
    def model_name(self) -> str:
        return f"Qwen3 ({self._base_model})"

    def save(self, path: pathlib.Path) -> None:
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

    def unload(self) -> None:
        if getattr(self, "model", None) is not None:
            del self.model
        gc.collect()
        torch.cuda.empty_cache()


class SimpleTeacher(Teacher):

    def chat(self, chat_history: ChatHistory) -> str:
        messages = [
            {"role": "user" if h.role == "Student" else "assistant", "content": h.content} for h in chat_history.root
        ]
        return self._llm.query(messages)

    def model_name(self) -> str:
        return self._llm.model_name


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
    elif backend == "google":
        llm = OpenAIAgent(model, client, temperature=0.15)
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
        teacher.llm().unload()

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
        student_llm.unload()

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

    clients = {
        "openai": OpenAI(),
        "ollama": Client(args.ollama_client),
        "google": OpenAI(
            api_key=os.getenv("GEMINI_API_KEY"),
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    }

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
        seed_llm.unload()
        seeds_path.write_text(seed_dataset.model_dump_json(indent=4))
        # Remove stale downstream outputs if seeds changed
        for f in output_dir.glob("int_*.json"):
            f.unlink()
        for f in output_dir.glob("eval_*.json"):
            f.unlink()

    for teacher in tqdm([
        # Teacher(resolve_llm(("ollama", "mistral-small3.1:24b"), clients)),
        # Teacher(resolve_llm(("ollama", "llama3.3:70b"), clients)),
        # Teacher(resolve_llm(("ollama", "gemma3:27b"), clients)),
        # Socratic(resolve_llm(("ollama", "eurecom-ds/phi-3-mini-4k-socratic"), clients)),
        # Teacher(resolve_llm(("openai", "gpt-4o"), clients)),
        # Teacher(resolve_llm(("google", "models/learnlm-2.0-flash-experimental"), clients)),
        SimpleTeacher(Qwen("/home/gatti/socratic-rl/trial-Z/train/policy_fn/"))
    ], desc="Teacher evaluation"):
        for max_interactions in tqdm([16, ], desc="Max interactions"):
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
                judge_llm.unload()
                evaluations_path.write_text(evaluations_dataset.model_dump_json(indent=4))

    print("Finished processing")
    exit(0)
