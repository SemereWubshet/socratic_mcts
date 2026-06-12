# Socratic Dialogue Generator

A pipeline for generating persona-driven Socratic student–teacher dialogues and benchmarking LLMs as Socratic teachers.

## Overview

AI tutors increasingly answer questions directly, which undermines students' ability to reason independently. The Socratic method — guiding students through discovery via open-ended questioning — is a powerful alternative, but training LLMs to use it requires high-quality dialogue data that is currently scarce.

This project addresses that gap in two ways:

- **Dataset generation**: A multi-agent pipeline that produces diverse Socratic conversations by pairing configurable student and teacher LLMs across varied student personas and topic seeds.
- **LLM benchmarking**: An evaluation harness that compares how well different LLMs perform as Socratic teachers, scored by a validated judge model.

The resulting dataset can be used to fine-tune educational LLMs in the Socratic teaching method.

For full details, see the [report](report.pdf).

## How It Works

1. **Seed generation** — `StudentSeed` samples chapters from the [Princeton TextbookChapters](https://huggingface.co/datasets/princeton-nlp/TextbookChapters) dataset and generates an opening question. Questions are either a straightforward inquiry about the chapter's main topics, or a misconception-laden claim designed to probe the teacher's ability to correct misunderstanding.

2. **Student simulation** — A `Student` agent responds in character as one of 7 distinct personas (see below), producing a realistic learning dynamic.

3. **Teacher response** — A `Teacher` (or `Socratic`) agent replies using Socratic principles: open-ended questions, indirect guidance, intellectual humility, and no direct answers.

4. **Judge evaluation** — A `Judge` agent (llama3.3:70b) assesses each completed conversation against a structured rubric covering topic coverage, Socratic adherence, and demonstrated student understanding.

## Student Personas

The pipeline simulates 7 student types to stress-test the teacher across different learning styles:

| Persona | Description |
|---|---|
| Effortless / Disengaged | Grasps concepts quickly but disengages when the topic feels insufficiently challenging |
| Curious but Tangential | Highly inquisitive, but curiosity leads down tangential paths away from the core topic |
| Easily Distracted | Enthusiastic but loses focus; needs redirection to the main objective |
| Overconfident | Learns quickly but overestimates understanding and dismisses foundational concepts |
| Error-Prone | Processes information fast but jumps to incorrect conclusions by overlooking nuance |
| Needs Examples | Learns best through concrete analogies; struggles with abstract concepts |
| Dependent | Eager to learn but relies heavily on guidance rather than independent reasoning |

## Installation

```bash
git clone https://github.com/<user>/socratic_mcts.git
cd socratic_mcts
pip install -r requirements.txt
```

For fine-tuning support, also install:

```bash
pip install -r requirements_peft.txt
```

## Usage

### Generate a Dataset

`rollout.py` runs the full pipeline — seed generation, student–teacher conversations, and judge evaluation — writing results to disk.

```bash
python src/rollout.py \
  --output-dir /path/to/output \
  --num-conversations 100 \
  --seed-llm ollama http://your-ollama-server:8080 mistral-nemo:12b-instruct-2407-fp16 \
  --student-llm ollama http://your-ollama-server:8080 mistral-nemo:12b-instruct-2407-fp16 \
  --teacher-llm ollama http://your-ollama-server:8080 mistral-nemo:12b-instruct-2407-fp16 \
  --judge-llm ollama http://your-ollama-server:8080 llama3.3:70b
```

The `--seed-llm`, `--student-llm`, `--teacher-llm`, and `--judge-llm` flags each accept a provider token (`ollama`, `openai`, or `google`), a base URL, and a model name.

### Benchmark LLMs as Socratic Teachers

`evaluate.py` sweeps across a set of teacher LLMs and interaction lengths, evaluating each configuration against the same seeds and student personas.

```bash
python src/evaluate.py \
  --root-dir /path/to/evaluation \
  --value-fn /path/to/evaluation/model \
  --num-conversation 200
```

### Visualize Results

`perf_eval.py` reads the evaluation output and produces charts broken down by model, model size, interaction length, opening question type, and student persona.

```bash
python src/perf_eval.py /path/to/eval_results /path/to/figs
```

Output figures are written as SVGs to the specified directory.

## Evaluation Results

The benchmark was run across six teacher LLMs at four interaction lengths (2, 4, 8, and 16 turns):

| Model | Type |
|---|---|
| `phi-3-mini-4k-socratic` | Fine-tuned baseline |
| `mistral-small3.1:24b` | Open-weight |
| `gemma3:27b` | Open-weight |
| `llama3.3:70b` | Open-weight |
| `gpt-4o` | Proprietary |
| `learnlm-2.0-flash-experimental` | Proprietary (education-focused) |

Detailed charts (success rate by persona, interaction length, model size, and opening question type) are available in `figs/`.

## Project Structure

```
socratic_mcts/
├── src/
│   ├── agents.py          # LLM agent classes (StudentSeed, Student, Teacher, Judge, ...)
│   ├── schemas.py         # Pydantic data models (Seed, Interaction, Evaluation)
│   ├── rollout.py         # Dataset generation script
│   ├── evaluate.py        # LLM benchmarking script
│   ├── perf_eval.py       # Performance visualization
│   ├── llm_comparison.py  # Cross-judge agreement (Cohen's kappa)
│   └── failure.py         # Failure pattern classification
├── templates/             # Prompt templates for each agent role
├── datasets/              # Generated seeds and evaluation results
├── figs/                  # Output figures from perf_eval.py
├── requirements.txt
└── requirements_peft.txt  # Additional deps for fine-tuning
```

## License

See [LICENSE](LICENSE).
