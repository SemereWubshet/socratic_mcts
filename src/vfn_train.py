import argparse
import json
import math
import pathlib
import shutil
from collections import defaultdict

import numpy as np
import ollama
import scipy
import torch
from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import Trainer
from transformers import TrainingArguments

from agents import OllamaAgent
from mcts import ValueFn
from rollout import gen_seeds, SeedDataset, InteractionDataset, \
    gen_teacher_student_interactions, EvaluationDataset, evaluate, ChatHistory
from tools import LLMAction


def discount_cumsum(x: np.ndarray, gamma: float) -> np.ndarray:
    """Calculates the discounted cumulative sum over a reward sequence `x`.

    y[t] - discount*y[t+1] = x[t]
    reversed(y)[t] - discount*reversed(y)[t-1] = reversed(x)[t]

    Args:
        gamma: The discount factor gamma.

    Returns:
        The sequence containing the discounted cumulative sums
        for each individual reward in `x` till the end of the trajectory.

     .. testcode::
        :skipif: True

        x = np.array([0.0, 1.0, 2.0, 3.0])
        gamma = 0.9
        discount_cumsum(x, gamma)

    .. testoutput::

        array([0.0 + 0.9*1.0 + 0.9^2*2.0 + 0.9^3*3.0,
               1.0 + 0.9*2.0 + 0.9^2*3.0,
               2.0 + 0.9*3.0,
               3.0])
    """
    return scipy.signal.lfilter([1], [1, float(-gamma)], x[::-1], axis=0)[::-1]


def vf_train(evaluation_dataset: EvaluationDataset,
             value_fn: ValueFn,
             train_dir: pathlib.Path,
             num_iterations: int,
             gamma: float = 1.,
             _lambda: float = 0.8) -> None:
    losses = []

    for it in tqdm(range(num_iterations)):
        dataset = defaultdict(list)

        for i in evaluation_dataset.root:
            assessment = i.assessment
            history = i.interaction.chat_history.root
            trajectory = [str(ChatHistory.model_validate(history[:2 * z + 1])) for z in
                          range(math.ceil(len(history) / 2))]
            values = [value_fn(h) for h in trajectory]

            # compute value targets
            vf_preds = np.array(values)
            rwd = np.zeros(vf_preds.shape[0])
            rwd[-1] = np.float32(1. if assessment else -1.)
            vpred_t = np.concatenate((vf_preds, [0.]))
            delta_t = -vpred_t[:-1] + rwd + gamma * vpred_t[1:]
            advantages = discount_cumsum(delta_t, gamma * _lambda)
            value_targets = advantages + vf_preds

            dataset["history"].extend(trajectory)
            dataset["labels"].extend(value_targets)

        hf_dataset = Dataset.from_dict(dataset)
        tokenized_dataset = hf_dataset.map(value_fn.batch_tokenize, batched=True, batch_size=8).shuffle()

        training_args = TrainingArguments(
            output_dir=str(train_dir / f"iteration-{it}"),
            num_train_epochs=1.,
            learning_rate=1e-5
        )
        trainer = Trainer(model=value_fn.model, args=training_args, train_dataset=tokenized_dataset)
        print(f"Starting training.... len={len(tokenized_dataset['labels'])}")
        result = trainer.train()

        losses.append({"iteration": it, "training_loss": result.training_loss})

        del trainer
        torch.cuda.empty_cache()

    (train_dir / "train_losses.json").write_text(json.dumps(losses))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="VALUE FN TRAIN", description="Train a teacher performance predictor for student-teacher interactions."
    )

    parser.add_argument("--output-dir", required=True, type=str, help="Path where to store pipeline outputs")
    parser.add_argument("--num-conversations", required=True, type=int, help="Number of conversations to generate")
    parser.add_argument("--max-interactions", default=15, type=int,
                        help="Maximum number of conversations rounds between the teacher and the student")
    parser.add_argument("--gamma", default=1., type=float, help="The discount factor")
    parser.add_argument("--lambda", dest="lambd", default=.8, type=float, help="The lambda trace")
    parser.add_argument("--num-iterations", default=5, type=int, help="Number of training iterations.")

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
        default=OllamaAgent(
            "llama3.3:70b",
            ollama.Client("http://atlas1api.eurecom.fr"), temperature=0., num_ctx=32_000
        )
    )
    parser.add_argument(
        "--predictor-model", type=str,
        help="Base encoder network to be used as predictor",
        default="allenai/longformer-base-4096"
    )
    parser.add_argument(
        "--gpu", type=str,
        help="GPU id to be used for training",
        default="cuda:0"
    )
    parser.add_argument(
        "--context-window", type=int,
        help="The context window to finetune the model",
        default=768
    )

    parser.add_argument("--use-cache", action="store_true", help="Don't run subprocess if output files exist")
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
            seed_dataset, args.student_llm, args.teacher_llm, max_interactions=args.max_interactions
        )
        interactions_path.write_text(interactions_dataset.model_dump_json(indent=4))
        evaluations_path.unlink(missing_ok=True)

    if evaluations_path.exists():
        print("Loading evaluation dataset", flush=True)
        evaluations_dataset = EvaluationDataset.model_validate_json(evaluations_path.read_text())
    else:
        print()
        print("Assessing teacher-student interactions", flush=True)
        evaluations_dataset = evaluate(interactions_dataset, args.judge_llm)
        evaluations_path.write_text(evaluations_dataset.model_dump_json(indent=4))

    print()
    print("Starting training....")
    value_fn = ValueFn(args.predictor_model, max_length=args.context_window, gpu=args.gpu)
    vf_train(evaluations_dataset, value_fn, output_dir / "train", args.num_iterations, args.gamma, args.lambd)

    print()
    print("Finished processing")

    exit(0)
