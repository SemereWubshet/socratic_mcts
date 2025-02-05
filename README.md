# Socratic Dialogue Generator

AI chatbots are increasingly being used in education. While these tools offer accessibility and a wealth of information, they often provide direct answers, hindering the development of students' critical thinking skills. The Socratic method, with its emphasis on guided discovery, offers a powerful alternative, but effectively replicating this method in AI requires substantial high-quality training data, which is currently scarce. This project introduces a pipeline for generating diverse, persona-driven Socratic Student-Teacher dialogues to address this data bottleneck. It also provides a benchmark for comparing the effectiveness of different LLMs as Socratic teachers, using a validated judge model (llama3.3:70b) to evaluate their generated conversations. This resource can be used to fine-tune educational LLMs in the Socratic teaching method.

### Generate Student-Teacher Interactions

The `rollout.py` script generates and evaluates Socratic conversations between a student and a teacher using LLMs. It supports using different LLMs for the Student, Teacher, and Judge models.

**Example Usage:**
```bash
python src/rollout.py \
  --output-dir /home/<user>/socratic_mcts/dataset \
  --num-conversations 100 \
  --seed-llm ollama http://your-ollama-server:8080 mistral-nemo:12b-instruct-2407-fp16 \
  --student-llm ollama http://your-ollama-server:8080 mistral-nemo:12b-instruct-2407-fp16 \
  --teacher-llm ollama http://your-ollama-server:8080 mistral-nemo:12b-instruct-2407-fp16 \
  --judge-llm ollama http://your-ollama-server:8080 llama3.3:70b
```

### Benchmark LLMs for Suitability as Socratic Teachers

The `evaluate.py` script allows you to benchmark and compare different LLMs for their effectiveness as Socratic teachers. It generates conversations using various teacher LLMs and evaluates their Socraticness using a llama3.3:70b as the judge LLM. This allows you to identify which LLMs are best suited for guiding students through Socratic dialogue.

**Example Usage:**
```bash
python src/evaluate.py \
--root-dir /home/<user>/socratic_mcts/evaluation \
--value-fn /home/<user>/socratic_mcts/evaluation/model \
--num-conversation 200
```
