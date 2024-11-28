import argparse
import itertools
import json
import pathlib
import random
from typing import Dict, List

import ollama

from src.conversation_generator import split_into_chunks

INTERACTION_TYPES = (
    "Demand deeper clarification about one of the major points on the topic.",
    "Request further explanations that go beyond the original text.",
    "Make misleading claims due to misunderstanding on one or more of the topics.",
    "Act confused about one of the major points, thus requiring further explanation from the teacher.",
    "Demonstrate inability to connect major points.",
    "Suggest a different understanding of a major point so to lead to a discussion about its validity.",
    "Request examples or applications of a major point in practical, real-world scenarios.",
    "Request the comparison to major points with similar or contrasting concepts.",
    "Pose \"what-if\" questions to explore the implications of the major point in various contexts.",
    "Question the foundational assumptions of the major point, prompting justification or re-explanation.",
    "Request an explanation of the major point in simpler terms or using analogies.",
    "Demonstrate understanding of some basic concepts but struggle to connect them to the broader major point.",
    "Ask questions that are tangentially related to the major point, requiring the teacher to refocus the conversation "
    "while addressing the inquiry.",
    "Ask for a detailed breakdown of a process or concept related to the major point.",
    "Ask if there are any arguments or evidence against the major point, prompting critical evaluation.",
    "Make overly broad statements about the major point, requiring clarification or correction.",
)


def gen_dataset(contexts: List[str]) -> List[Dict[str, str]]:
    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")

    base_prompt = pathlib.Path("./templates/seed.txt").read_text(encoding="UTF-8")

    dataset = []

    for context in contexts:
        single_liner = context.replace("\n", " ")
        interaction_type = random.choice(INTERACTION_TYPES)
        content = base_prompt.format(context=single_liner, interaction_type=interaction_type)

        response = client.chat(model="mistral-nemo:12b-instruct-2407-fp16",
                               messages=[{"role": "user", "content": content}])
        opening: str = response["message"]["content"]
        opening = opening.strip().replace("\"", "")
        dataset.append({"seed": context, "interaction_type": interaction_type, "opening": opening})

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", required=True, help="Input textual data on covered topics",
                        type=argparse.FileType("r"))
    parser.add_argument("-o", "--output", required=True, help="Output json dataset with generate seed interactions",
                        type=argparse.FileType("w"))
    parser.add_argument("-n", "--num-examples", default=50, help="Number of output examples to produce",
                        type=int)

    args = parser.parse_args()

    docs = args.input.read()
    topics = docs.split("Title: ")  # To avoid having chunks with two distinct subjects
    chunks = list(itertools.chain.from_iterable(split_into_chunks(topic, chunk_size=2000) for topic in topics))
    selected = random.choices(chunks, k=args.num_examples)

    data = gen_dataset(selected)

    args.output.write(json.dumps(data, indent=4))
