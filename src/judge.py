import json
import pathlib
from typing import Dict, List, Any

import ollama

from src.conversation_generator import ChatHistory


def gen_dataset(conversations: List[ChatHistory]) -> List[Dict[str, Any]]:
    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")

    system_prompt = pathlib.Path("./templates/judge.txt").read_text(encoding="UTF-8")
    answer_prompt = pathlib.Path("./templates/answer.txt").read_text(encoding="UTF-8")

    dataset = []

    for conversation in conversations:
        text_chunk = conversation.get_text_chunk()
        seed_question = conversation.get_seed()

        content = answer_prompt.format(context=text_chunk, question=seed_question)

        response = client.chat(model="mistral-nemo:12b-instruct-2407-fp16",
                               messages=[{"role": "user", "content": content}],
                               options={
                                   "num_ctx": 16_000,
                                   "temperature": 0.1,
                               })

        topics = response["message"]["content"]

        eval_query = f"# Main topics\n{topics}\n\n# Chat history\n{conversation}"
        response = client.chat(model="mistral-nemo:12b-instruct-2407-fp16",
                               messages=[
                                   {"role": "system", "content": system_prompt},
                                   {"role": "user", "content": eval_query}],
                               options={
                                   "num_ctx": 32_000,
                                   "temperature": 0.1,
                               })

        evaluation: str = response["message"]["content"]
        as_json = json.loads(evaluation)
        dataset.append({"topics": topics,
                        "history": conversation,
                        "reason": as_json["feedback"],
                        "assessment": as_json["assessment"]})

    return dataset


if __name__ == "__main__":
    with open("caches/conversations.json", "r") as f:
        data = json.load(f)

    hist = [ChatHistory.from_history(c) for c in data]

    new_data = gen_dataset(hist)
    new_new_data = []
    for n in new_data:
        new_new_data.append({"history": n["history"].history, "reason": n["reason"], "assessment": n["assessment"]})


    # history = new_data[0].get_history_list()
    with open("datasets/new_dataset.json", "w") as f:
        json.dump(new_new_data, f, indent=4)

    pass
