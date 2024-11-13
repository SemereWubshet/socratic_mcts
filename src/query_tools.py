import openai
import re
from openai import OpenAI
client = OpenAI()

def create_prompt(content):
    prompt = (
        "You are a curious student with some foundational knowledge across general topics and a "
        "strong desire to learn more. Given the topic below, ask a question that reflects your "
        "curiosity—one that seeks to understand concepts, explore ideas, or uncover reasoning "
        "behind the subject matter. Your question should show interest in learning further "
        "without needing excessive detail.\n"
        "Topic:\n"
        f"{content}\n"
        "Please generate a list of 3 questions following this guidance."
    )
    return prompt


def generate_socratic_questions(content):
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a curious student with some foundational knowledge across general topics and a "
        "strong desire to learn more. Given the topic below, ask a question that reflects your "
        "curiosity—one that seeks to understand concepts, explore ideas, or uncover reasoning "
        "behind the subject matter. Your question should show interest in learning further "
        "without needing excessive detail. Please generate a list of 3 questions following "
        "this guidance about the topic given below."},
            {"role": "user", "content": f"{content}"}
        ]
    )
    questions = response.choices[0].message
    questions = re.findall(r'\d+\.\s(.*?\?)', str(questions))
    questions = '\n'.join(questions)
    return questions
