import ollama
import openai
import re
from openai import OpenAI



with open('../templates/query_llm.txt', 'w') as f:
    query_role = f.read()
print(query_role)

def openai_gen_soc_questions(content):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": query_role},
            {"role": "user", "content": content}
        ]
    )
    questions = response.choices[0].message
    questions = re.findall(r'\d+\.\s(.*?\?)', str(questions))
    questions = '\n'.join(questions)
    return questions

def ollama_gen_soc_questions(content):
    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")
    response = client.chat(model="llama3.1", messages=[{"role": "system", "content": query_role},
                                                       {"role": "user", "content": content}])
    # print(response["message"]["content"])

    questions = response["message"]["content"]
    # questions = re.findall(r'\d+\.\s(.*?\?)', questions)
    return questions

# x = "The basic mammalian body type is quadrupedal, with most mammals using four limbs for terrestrial locomotion; but in some, the limbs are adapted for life at sea, in the air, in trees or underground. The bipeds have adapted to move using only the two lower limbs, while the rear limbs of cetaceans and the sea cows are mere internal vestiges. Mammals range in size from the 30–40 millimetres (1.2–1.6 in) bumblebee bat to the 30 metres (98 ft) blue whale—possibly the largest animal to have ever lived. Maximum lifespan varies from two years for the shrew to 211 years for the bowhead whale. All modern mammals give birth to live young, except the five species of monotremes, which lay eggs. The most species-rich group is the viviparous placental mammals, so named for the temporary organ (placenta) used by offspring to draw nutrition from the mother during gestation."
#
# out = ollama_gen_soc_questions(x)
# print(type(out))
# print(out)