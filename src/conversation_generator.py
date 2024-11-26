import argparse
import json
import random
from typing import TextIO
import query_tools as qt


# Remove this function: I can just qt.ollama_gen_seed() directly
def gen_seed_question(text_chunk:str) -> str:
    """Call to llm to generate a specific seed topic given a chunk"""
    seed_question = qt.ollama_gen_soc_question(text_chunk)
    return seed_question


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
        if self.history[-1]['role'] != 'evaluation' and self.history[-1]['role'] != 'text_chunk':
            role = self.history[-1]['role']
            query = self.history[-1]['query']
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
    teacher_response = qt.ollama_gen_teacher_response(str(history))
    return teacher_response


def student(text_chunk:str, seed_question:str, history:ChatHistory) -> str:
    """Generate student response based on seed and history"""
    student_response = qt.ollama_gen_student_response(text_chunk, seed_question, str(history))
    return student_response


def judge(seed_topic:str, text_chunk:str, history:ChatHistory) -> int:
    """Judge whether the teacher displayed correct Socratic behavior"""
    judge_response = qt.ollama_judge(seed_topic, text_chunk, str(history))
    return judge_response

def generate_exchange(text_chunk:str) -> ChatHistory:
    """Generate Socratic dialogue between a student and a teacher"""
    seed_question = gen_seed_question(text_chunk)
    history = ChatHistory()
    history.add_text_chunk(text_chunk)
    history.add_student(seed_question)

    for _ in range(depth - 1):
        teacher_query = teacher(history)
        history.add_teacher(teacher_query)

        student_query = student(text_chunk, seed_question, history)
        history.add_student(student_query)

    teacher_query = teacher(history)
    history.add_teacher(teacher_query)

    return history

def split_into_chunks(text, chunk_size):
    """Split a given text file into manageable pieces"""
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    random.shuffle(chunks) # Randomize chunks
    return chunks

def pipeline(input_name:TextIO, output_name:TextIO, number_of_conversations) -> None:
    """Assemble tools to build a Socratic pedagogical dialogue"""
    contents = input_name.read()
    text_chunks = split_into_chunks(contents, chunk_size)

    exchanges = []

    for index in range(number_of_conversations):
        text_chunk = text_chunks[index]
        exchange = generate_exchange(text_chunk)
        exchanges.append(exchange)


    exchanges_dump = [history.get_history_list() for history in exchanges]
    json.dump(exchanges_dump, output_name, indent=4) #args.o words in output_name's place for some reason
    # args.o.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', required=True, help='', type=argparse.FileType('r'))
    parser.add_argument('-o', required=True, help='', type=argparse.FileType('w'))

    # Attributes of socratic conversations
    depth = 5 # Depth of conversations
    chunk_size = 1000 # Chunk size of splits in input file
    num_conversations = 10 # Number of conversations
    args = parser.parse_args()

    # Run pipeline
    pipeline(args.i, args.o, num_conversations)


    """Tests
    
    # j = [{'role': 'text_chunk', 'query': 'The sky is blue because of things.'},
    #      {'role': 'student', 'query': 'Why is the sky blue?'},
    #      {'role': 'teacher', 'query': 'Could it be the angle of the sun?'},
    #      {'role': 'student', 'query': 'Perhaps the blue light is spread by the atmosphere giving it a blue tint.'},
    #      {'role': 'teacher', 'query': "Exactly! That's also the reason the sky turns orange during sunrise and sunset."}]
    # k = [{'role': 'text_chunk', 'query': 'Pineapple.'}, {'role': 'student', 'query': 'Why fruit?'},
    #      {'role': 'teacher', 'query': 'Delicious, no?'}, {'role': 'student', 'query': 'True!'},
    #      {'role': 'teacher', 'query': "Amen."}]
    #
    # l = ChatHistory.from_history(j)
    # m = ChatHistory.from_history(k)
    #
    # print(l.get_history_list())
    # print(l.get_eval())
    # print(str(l))
    #
    # n = [o.get_history_list() for o in [l, m]]
    # json.dump(n, args.o, indent=4)
    
    """