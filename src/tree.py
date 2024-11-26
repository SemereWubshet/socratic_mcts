import argparse
import random
import json
import pathlib
import copy
from typing import TextIO, List, Optional, Any

# from pydantic import BaseModel

import query_tools as qt


def gen_seed_topic(text_chunk:str) -> str:
    """Call to llm to generate a specific seed topic given a chunk"""
    seed_topic = qt.ollama_gen_seed(text_chunk)
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
        query = qt.ollama_gen_teacher_response(self.history())
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
    score:int
    def __init__(self, reply:str, parent:StudentNode):
        self.reply = reply
        self.parent = parent
        self.score = 0
        self.children = []
    #     # super().__init__(reply=self.reply, parent=parent, children=self.children)

    def history(self) -> str:
        # Traverse the tree and generate the history
        node_str = f"Teacher: {self.reply}\n"
        parent_str = self.parent.history()
        history_str = parent_str + "\n" + node_str
        return history_str

    def query(self) -> StudentNode:
        query = qt.ollama_gen_student_response(self.get_seed(), self.history())
        student_node = StudentNode(parent=self, question=query)
        self.children.append(student_node)
        return student_node

    def to_dict(self) -> dict[str, Any]:
        return {"role": "teacher", "reply": self.reply, "score": self.score, "children": [child.to_dict() for child in self.children]}

    def get_seed(self) -> str:
        seed = self.parent.get_seed()
        return seed

    @classmethod
    def from_dict(cls, d: dict[str,Any], parent: StudentNode) -> "TeacherNode":
        assert d["role"] == 'teacher'
        reply = d["reply"]
        score = d["score"]

        teacher_node = TeacherNode(reply, parent)
        teacher_node.score = score
        children = [StudentNode.from_dict(child, teacher_node) for child in d["children"]]
        teacher_node.children.extend(children)
        return teacher_node

def split_into_chunks(text, chunk_size):
    """Split a given text file into manageable pieces"""
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    random.shuffle(chunks) # Randomize chunks
    return chunks


def gen_tree(student_node:StudentNode, tree_width:int, tree_depth:int) -> None:
    if tree_depth == 0:
        return None
    for width in range(tree_width):
        teacher_node = student_node.query()
        # Score teacher reply and save it here
        for width in range(tree_width):
            student_node = teacher_node.query()
            gen_tree(student_node, tree_width, tree_depth-1)

def pipeline(input_name:TextIO, output_name:TextIO, num_trees:int, tree_width:int, tree_depth:int) -> None:
    """Assemble tools to build Socratic pedagogical dialogue trees"""
    contents = input_name.read()
    text_chunks = split_into_chunks(contents, chunk_size)
    tree_collection = []
    tree_collection_dump = []
    for index in range(num_trees):
        seed = gen_seed_topic(text_chunks[index])
        student_root_node = StudentRootNode(text_chunks[index], seed)
        gen_tree(student_root_node, tree_width, tree_depth)

        tree_collection.append(student_root_node)
        tree_collection_dump.append(student_root_node.to_dict())

    json.dump(tree_collection_dump, output_name, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, help='', type=argparse.FileType('r'))
    parser.add_argument('-o', required=True, help='', type=argparse.FileType('w'))

    args = parser.parse_args()

    # Pipeline parameters
    num_trees = 1 # Number of trees generated
    tree_width = 1 # Width of  conversation tree
    tree_depth = 1 # Depth of conversation tree
    chunk_size = 5000 # Chunk size of splits in the input file

    # pipeline(args.i, args.o, num_trees, tree_width, tree_depth)

    chunk = "In the Middle Ages, liberal arts were taught in European universities as part of the Trivium, an introductory curriculum involving grammar, rhetoric, and logic, and of the Quadrivium, a curriculum involving the \"mathematical arts\" of arithmetic, geometry, music, and astronomy."
    student0 = StudentRootNode(chunk, "What types of liberal studies did they have in the middle ages?")

    gen_tree(student0, 2, 2)
    cake = student0.to_dict()
    print(cake)
    json.dump([cake, ], args.o, indent=4)
    args.o.close()