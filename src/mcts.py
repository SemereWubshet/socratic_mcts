import random
from typing import Optional, Tuple, Callable, List

import numpy as np
import ollama
import scipy
import torch
from datasets import load_dataset, tqdm, Dataset
from transformers import AutoTokenizer, ModernBertForSequenceClassification, Trainer
from transformers import TrainingArguments

from agents import Student, Teacher, Judge, OllamaAgent, LLM
from rollout import gen_seeds, ChatHistory, Message, Seed


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


class SeededJudge:

    def __init__(self, seed: Seed, judge: Judge):
        self._seed = seed
        self._judge = judge

    def evaluate(self, chat_history: str) -> Tuple[str, str]:
        return self._judge.evaluate(self._seed.main_topics, chat_history)


class ValueFn:

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
        self.model = ModernBertForSequenceClassification.from_pretrained("answerdotai/ModernBERT-base",
                                                                         num_labels=1).to("cuda:0")

    def __call__(self, history: str) -> float:
        _history = self.tokenizer(history, return_tensors="pt").to("cuda:0")
        with torch.no_grad():
            value = float(self.model(**_history).logits)
        return value


class StudentNode:

    def __init__(self, question: str, parent: Optional["TeacherNode"] = None, end: bool = False):
        self.question = question
        self.parent = parent
        self.end = end
        self.children: List[TeacherNode] = []

        # MCTS
        self.v: float = 0.
        self.terminal: bool = False

    def expand(self, teacher: Teacher) -> "TeacherNode":
        assert not self.end
        history = self.history()
        reply = teacher.chat(history)
        teacher_node = TeacherNode(reply, self)
        self.children.append(teacher_node)
        return teacher_node

    def history(self) -> ChatHistory:
        history = ChatHistory([]) if self.parent is None else self.parent.history()
        history.root.append(Message(role="Student", content=self.question, end=self.end))
        return history

    def trajectory(self) -> List["StudentNode"]:
        """
        Returns a list of all parent Student nodes ending at this node.
        The list is ordered from root to leaf node.
        """
        if self.parent is None:
            return [self]

        return self.parent.parent.trajectory() + [self, ]

    def depth(self) -> int:
        if self.parent is None:
            return 0
        return self.parent.depth() + 1

    def get_eligible(self, eligibility_fn: Callable[["StudentNode"], bool]) -> List["StudentNode"]:
        output = []

        for child in self.children:
            eligible = child.child.get_eligible(eligibility_fn)
            output.extend(eligible)

        if eligibility_fn(self):
            output.append(self)

        return output


class TeacherNode:

    def __init__(self, reply: str, parent: StudentNode):
        self.reply = reply
        self.parent = parent
        self.child: Optional[StudentNode] = None

        # MCTS
        self.n: int = 0
        self.w: float = 0.
        self.q: float = 0.

    def expand(self, student: Student) -> "StudentNode":
        assert self.child is None
        history = self.history()
        question, end = student.chat(history)
        student_node = StudentNode(question, self, end)
        self.child = student_node
        return student_node

    def history(self) -> ChatHistory:
        history = self.parent.history()
        history.root.append(Message(role="Teacher", content=self.reply, end=False))
        return history

    def depth(self) -> int:
        return self.parent.depth()


def select(root: StudentNode) -> TeacherNode:
    ni = np.array([c.n for c in root.children])
    N = np.sum(ni)
    terminal_states = np.array([c.child is not None and c.child.terminal for c in root.children])

    q = np.array([c.q for c in root.children])
    q[terminal_states] = -np.inf

    u = np.sqrt(0.1 * np.log(1 + N) / (1 + ni))
    idx = int(np.argmax(q + u))
    selected = root.children[idx]
    student_node = selected.child

    if student_node is None:
        return selected

    return select(student_node)


def expand(node: TeacherNode,
           student: Student,
           teacher: Teacher,
           value_fn: ValueFn,
           judge: SeededJudge,
           max_depth: int = 15) -> Tuple[bool, StudentNode]:
    student_node = node.expand(student)

    if student_node.end or student_node.depth() >= max_depth:
        chat_history = str(student_node.history())
        _, assessment = judge.evaluate(chat_history)
        student_node.v = 1. if assessment else -1.
        student_node.terminal = True
        return True, student_node

    student_node.v = value_fn(str(student_node.history()))

    student_node.terminal = False
    student_node.expand(teacher)
    student_node.expand(teacher)
    student_node.expand(teacher)

    return False, student_node


def backup(leaf: StudentNode) -> None:
    teacher_node: TeacherNode = leaf.parent
    while teacher_node is not None:
        teacher_node.n += 1
        teacher_node.w += leaf.v
        teacher_node.q = teacher_node.w / teacher_node.n
        teacher_node = teacher_node.parent.parent


def mcts_train(seed_llm: LLM,
               student_llm: LLM,
               teacher_llm: LLM,
               judge_llm: LLM,
               num_of_conversations: int,
               train_iterations: int,
               max_depth: int = 15,
               gamma: float = 1.,
               _lambda: float = 0.8) -> None:
    wikipedia = load_dataset("wikimedia/wikipedia", "20231101.simple")

    judge = Judge(judge_llm)
    teacher = Teacher(teacher_llm)
    value_fn = ValueFn()

    for _ in tqdm(range(train_iterations)):
        seed_dataset = gen_seeds(wikipedia, seed_llm, num_of_conversations)

        dataset = {"history": [], "value_target": []}

        for seed in tqdm(seed_dataset.root):
            seeded_judge = SeededJudge(seed, judge)
            student_type = random.randint(0, len(Student.TYPES) - 1)
            student = Student(student_llm, seed.main_topics, student_type)

            root = StudentNode(seed.question)
            root.v = value_fn(str(root))
            root.expand(teacher)
            root.expand(teacher)
            root.expand(teacher)

            leaf = root
            ended = False
            while not ended:
                selected = select(root)
                ended, leaf = expand(selected, student, teacher, value_fn, seeded_judge, max_depth)
                backup(leaf)

            # compute value targets
            trajectory = leaf.trajectory()
            vf_preds = np.array([t.v for t in trajectory[:-1]])
            rwd = np.zeros(vf_preds.shape[0])
            rwd[-1] = np.float32(leaf.v)
            vpred_t = np.concatenate((vf_preds, [0.]))
            delta_t = -vpred_t[:-1] + rwd + gamma * vpred_t[1:]
            advantages = discount_cumsum(delta_t, gamma * _lambda)
            value_targets = advantages + vf_preds

            for node, value_target in zip(trajectory, value_targets):
                dataset["history"].append(str(node.history()))
                dataset["value_target"].append(value_target)

            def tokenize_function(examples):
                return value_fn.tokenizer(
                    # TODO: figure out the max_length: run one or two examples and check the max_length of tokens
                    #       np.sum(np.array(tokenized_dataset["input_ids"][X]) != 50283)
                    examples["history"], padding='max_length', truncation=True, max_length=1024
                ).to("cuda:0")

            hf_dataset = Dataset.from_dict(dataset)
            hf_dataset = hf_dataset.rename_column("value_target", "labels")
            tokenized_dataset = hf_dataset.map(tokenize_function, batched=True).shuffle()

            training_args = TrainingArguments(output_dir="test_trainer", num_train_epochs=1.)
            trainer = Trainer(model=value_fn.model, args=training_args, train_dataset=tokenized_dataset)
            trainer.train()

            del trainer
            torch.cuda.empty_cache()


if __name__ == "__main__":
    model = "mistral-nemo:12b-instruct-2407-fp16"
    llm = OllamaAgent(model=model, client=ollama.Client("http://atlas1api.eurecom.fr:8019"), temperature=0.)

    model = "mistral-nemo:12b-instruct-2407-fp16"
    teacher_llm = OllamaAgent(model=model, client=ollama.Client("http://atlas1api.eurecom.fr:8019"), temperature=0.7)

    model = "llama3.3:70b"
    judge_llm = OllamaAgent(model=model, client=ollama.Client("http://atlas1api.eurecom.fr:8019"), temperature=0.)
    # TODO: num_of_conversations ~60-70 (start with 15 x 5 training iterations)
    mcts_train(llm, llm, teacher_llm, judge_llm, 1, 1, max_depth=3)
