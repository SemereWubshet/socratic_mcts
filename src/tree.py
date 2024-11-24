import argparse
import json
import pathlib
import copy
from typing import TextIO, List, Optional, Any

# from pydantic import BaseModel

import query_tools as qt
def gen_seed_topic(text_chunk:str) -> str:
    """Call to llm to generate a specific seed topic given a chunk"""
    seed_topic = qt.openai_gen_seed(text_chunk)
    return seed_topic

class StudentNode:
    children:List["TeacherNode"]
    question:str
    parent:Optional["TeacherNode"]
    def __init__(self, parent:Optional["TeacherNode"], question: str):
        self.parent = parent
        self.question = question
        self.children = []

    def query(self) -> "TeacherNode":
        query = qt.openai_gen_teacher_response(self.history())
        teacher_node = TeacherNode(reply=query, parent=self)
        self.children.append(teacher_node)
        return teacher_node

    def get_seed(self) -> str:
        seed = self.parent.get_seed()
        return seed
    def history(self) -> str:
        # Traverse the tree and generate the history
        node_str = f"Student: {self.question}\n"
        parent_str = self.parent.history()
        history_str = parent_str + "\n" + node_str
        return history_str

    def to_dict(self) -> dict[str, Any]:
        return {'role': "student", "question": self.question,
                   "children": [child.to_dict() for child in self.children]}
    @classmethod
    def from_dict(cls, d: dict[str,Any], parent: Optional["TeacherNode"]) -> "StudentNode":
        assert d["role"] == 'student'
        question = d["question"]
        student_node = StudentNode(parent, question)
        children = [TeacherNode.from_dict(child, student_node) for child in d["children"]]
        student_node.children.extend(children)
        return student_node

class StudentRootNode(StudentNode):
    seed:str
    text_chunk:str

    def __init__(self, text_chunk:str, question: str):
        super().__init__(parent=None, question=question)
        self.seed = question
        self.text_chunk = text_chunk
        self.question = question
        self.children = []

    def history(self) -> str:
        # Traverse the tree and generate the history
        history_str = f"Student: {self.question}\n"
        return history_str

    def get_seed(self) -> str:
        return self.seed

    def to_dict(self) -> dict[str, Any]:
        return {"text_chunk": self.text_chunk, "role": "student",
                "question": self.question, "children": [child.to_dict() for child in self.children]}

    @classmethod
    def from_dict(cls, d: dict[str,Any], parent: Optional["TeacherNode"]) -> "StudentRootNode":
        assert "text_chunk" in d
        text_chunk = d["text_chunk"]
        question = d["question"]
        student_root_node = StudentRootNode(text_chunk, question)
        children = [TeacherNode.from_dict(child, student_root_node) for child in d["children"]]
        student_root_node.children.extend(children)
        return student_root_node

class TeacherNode:
    children:List[StudentNode]
    parent:StudentNode
    reply:str
    def __init__(self, reply:str, parent:StudentNode):
        self.reply = reply
        self.parent = parent
        self.children = []
    #     # super().__init__(reply=self.reply, parent=parent, children=self.children)

    def history(self) -> str:
        # Traverse the tree and generate the history
        node_str = f"Teacher: {self.reply}\n"
        parent_str = self.parent.history()
        history_str = parent_str + "\n" + node_str
        return history_str
    def query(self) -> StudentNode:
        query = qt.openai_gen_student_response(self.get_seed(), self.history())
        student_node = StudentNode(parent=self, question=query)
        self.children.append(student_node)
        return student_node

    def to_dict(self) -> dict[str, Any]:
        return {"role": "teacher", "reply": self.reply, "children": [child.to_dict() for child in self.children]}

    def get_seed(self) -> str:
        seed = self.parent.get_seed()
        return seed

    @classmethod
    def from_dict(cls, d: dict[str,Any], parent: StudentNode) -> "TeacherNode":
        assert d["role"] == 'teacher'
        reply = d["reply"]

        teacher_node = TeacherNode(reply, parent)
        children = [StudentNode.from_dict(child, teacher_node) for child in d["children"]]
        teacher_node.children.extend(children)
        return teacher_node





























class ChatHistory:
    """The history will store a sequence of student and teacher queries."""
    def __init__(self):
        self.history = []

    # Add at the beginning
    def add_text_chunk(self, query:str) -> None:
        self.history.append({'role': 'text_chunk', 'query': query})
    def add_student(self, query:str) -> None:
        self.history.append({'role': 'student', 'query': query})
    def add_teacher(self, query:str) -> None:
        self.history.append({'role': 'teacher', 'query': query})

    # Add at the end
    def add_eval(self, query:int) -> None:
        self.history.append({'role': 'evaluation', 'query': query})
    def get_history_list(self) -> list:
        return self.history
    def __str__(self) -> str:
        history_str = []
        for entry in self.history[1:-1]:
            role = entry['role']
            query = entry['query']
            history_str.append(f"{role.capitalize()}: {query}")
        return "\n".join(history_str)
    def is_empty(self) -> bool:
        return len(self.history) == 0
    def get_text_chunk(self) -> str:
        return self.history[0]["query"]
    def get_seed(self) -> str:
        return self.history[1]["query"]
    def get_eval(self) -> str:
        if self.history[-1]['role'] == 'evaluation':
            return self.history[-1]["query"]
        else:
            return ""

    @classmethod
    def from_history(cls, exchanges: list) -> 'ChatHistory':
        """Regenerate a ChatHistory object from a string representation of exchanges."""
        chat_history = cls()  # Create a new instance of ChatHistory

        for entry in exchanges:
            if entry['role'] == 'text_chunk':
                chat_history.add_text_chunk(entry['query'])
            elif entry['role'] == 'student':
                chat_history.add_student(entry['query'])
            elif entry['role'] == 'teacher':
                chat_history.add_teacher(entry['query'])
            elif entry['role'] == 'evaluation':
                chat_history.add_eval(entry['query'])

        return chat_history

def teacher(history:ChatHistory) -> str:
    """Generate teacher response based on history"""
    teacher_response = qt.openai_gen_teacher_response(str(history))
    return teacher_response

def student(seed:str, history:ChatHistory) -> str:
    """Generate student response based on seed and history"""
    if history.is_empty():
        student_response = qt.openai_gen_soc_question(seed)
    else:
        student_response = qt.openai_gen_student_response(seed, str(history))

    return student_response


def judge(history:ChatHistory) -> int:
    """Judge whether the teacher displayed correct Socratic behavior"""
    seed = history.get_seed()
    text_chunk = history.get_text_chunk()
    history_str = str(history)
    judge_response = qt.openai_gen_judge(seed, text_chunk, history_str) # Using open_ai at the moment
    return judge_response

def generate_exchanges(seed:str, text_chunk:str, history:ChatHistory, tree_width:int, tree_depth:int) -> list:
    """Generate iter Socratic responses by the teacher"""
    if tree_depth == 0:
        return []

    student_query = student(seed, history)

    exchanges = []
    for _ in range(tree_width): # Multiple teacher responses
        new_history = copy.deepcopy(history)
        new_history.add_student(student_query)  # Student response

        teacher_query = teacher(new_history)  # Teacher response
        new_history.add_teacher(teacher_query)
        result = judge(seed, text_chunk, new_history) # Judge verdict

        # Additional to global variables
        tree_list.append(new_history)
        results_list.append(result)

        item_histories = generate_exchanges(seed, text_chunk, new_history, tree_width, tree_depth - 1)
        exchanges.append({'history': new_history, 'result': result, 'children': item_histories}) # Nested dictionary

    return exchanges

def split_into_chunks(text, chunk_size):
    """Split a given text file into manageable pieces"""
    # Split text into chunks of size chunk_size
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

def pipeline(input_name:TextIO, output_name:TextIO, tree_width:int, tree_depth:int) -> None:
    """Assemble tools to build a Socratic pedagogical dialogue"""
    contents = input_name.read()
    text_chunks = split_into_chunks(contents, chunk_size)
    master_collection = []
    for text_chunk in text_chunks:
        seed = gen_seed_topic(text_chunk)
        history = ChatHistory()
        tree_dict = generate_exchanges(seed, text_chunk, history, tree_width, tree_depth)
        master_collection.append(tree_dict)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, help='', type=argparse.FileType('r'))
    parser.add_argument('-o', required=True, help='', type=argparse.FileType('w'))

    args = parser.parse_args()

    # Pipeline parameters
    tree_width = 1 # Width of  conversation tree
    tree_depth = 1 # Depth of conversation tree
    chunk_size = 80000 # Chunk size of splits in the input file

    tree_list = [] # Global list of conversations
    results_list = [] # Global list of conversations
    pipeline(args.i, args.o, tree_width, tree_depth)

    tree_dump = [history.get_history() for history in tree_list]
    print(tree_dump)
    json.dump(tree_dump, args.o, indent=4)

    for result in results_list:
        with open('datasets/' + 'results.txt', 'w') as f:
            f.write("\n ===== " + result)