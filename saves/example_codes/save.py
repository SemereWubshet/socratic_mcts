import requests
from bs4 import BeautifulSoup

# Locate the content div
content_div = soup.find('div', class_='mw-parser-output')
print(content_div)

# Extract all paragraphs
paragraphs = content_div.find_all('p')

# Join the paragraph text
page_text = "\n".join([para.get_text() for para in paragraphs if para.get_text()])
print(page_text)

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


text_chunk = "Natural selection is the differential survival and reproduction of individuals due to differences in phenotype. It is a key mechanism of evolution, the change in the heritable traits characteristic of a population over generations. Charles Darwin popularised the term natural selection, contrasting it with artificial selection, which is intentional, whereas natural selection is not."
seed = "How does natural selection influence the development and complexity of an organism's features, rather than just favoring existing traits?"
history_str = ("student: How does natural selection influence the development and complexity of an organism's features, rather than just favoring existing traits?"
           "teacher: That's an interesting question. If we consider natural selection as a process that favors traits aiding survival, what do you think might drive the appearance of entirely new traits or features in an organism, rather than just reinforcing existing ones?"
           "student: Hmm, maybe environmental changes or challenges? Could they create the need for new traits, leading to the development of more complex features?"
           "teacher: Good thought! If environmental challenges influence the need for new traits, how might these traits arise? Do you think it's possible for new features to emerge through random mutations or changes over generations?"
           "student: Ah, I see! Mutations might introduce new features, and natural selection could then favor those that are advantageous for survival. So, natural selection isn't just about existing traits—it could guide the development of entirely new ones based on changing needs?"
           "teacher: Exactly! You've made a key insight. Natural selection can not only refine existing traits but also act on new mutations that arise, which can lead to the development of more complex features over time. How do you think this process might unfold in the long term, shaping the complexity of organisms?"
           "student: It seems like, over time, these small changes and mutations could accumulate, leading to significant new features that help the organism survive better in its environment. So, it's a combination of mutation and selection that drives complexity, right?"
           "teacher: Yes, you've got it! As mutations occur, natural selection favors those that improve survival, which can lead to the development of increasingly complex features. So, it’s not just about survival of the fittest, but also the creative potential of new traits that emerge through mutation and are selected for by the environment.")
out = ollama_judge(seed, text_chunk, history_str)
print(out)
print('hello')


[
    [
        {
            "role": "student",
            "query": "How do the physical adaptations of amphibian larvae, such as gills and tail fins, help them transition into air-breathing adults with lungs?"
        },
        {
            "role": "teacher",
            "query": "What exactly do you think \"transitioning\" means in this context? Are we talking about a complete change from one form to another, or is there some continuity between the larval stage and adulthood?"
        },
        {
            "role": "student",
            "query": "I think transitioning refers to the process of adapting from one life stage to another, but not necessarily a complete transformation. The physical adaptations of amphibian larvae, such as gills for respiration and tail fins for swimming, are crucial for their survival in aquatic environments.\n\nHowever, if we consider the fact that these larvae eventually develop into air-breathing adults with lungs, it suggests that there's some continuity between the two stages. For example, both larval and adult amphibians have a similar body structure, such as four limbs, which indicates a shared ancestry.\n\nIn this sense, I think transitioning refers to the transformation of an aquatic, gill-breathing tadpole into a semi-aquatic or terrestrial air-breathing frog, while still retaining some characteristics from its earlier stage. The physical adaptations of the larval stage serve as a bridge, allowing them to survive in their initial environment before developing further into adults with lungs.\n\nThis continuity between stages makes me wonder if it's possible that there are other animals, besides amphibians, that undergo similar transformations from one life stage to another, while still retaining some characteristics from their earlier stage."
        },
        {
            "role": "teacher",
            "query": "So you're thinking that the physical adaptations of amphibian larvae act as a kind of intermediate form, allowing them to survive in an aquatic environment before eventually developing into air-breathing adults with lungs. And this got you wondering if there are other animals that undergo similar transformations.\n\nWhat do you think might be some key factors that would allow for such continuity between life stages? For example, is it the presence of a certain type of environment, like water or air, that influences the development of these intermediate forms? Or is it something more fundamental to the biology of these organisms themselves?"
        },
        {
            "role": "student",
            "query": "I think one key factor that allows for continuity between life stages is the idea of \"preadaptation\". This means that some characteristics from earlier life stages can be repurposed or modified to serve new functions in later stages. In the case of amphibian larvae, their gills and tail fins are preadaptations for their eventual transition to air-breathing adults with lungs.\n\nAdditionally, I believe that developmental plasticity also plays a crucial role. This refers to the ability of an organism's development to be influenced by its environment, allowing it to adapt to changing conditions. For example, amphibian larvae can grow and develop in response to changes in their aquatic environment, such as water depth or temperature.\n\nIt's also possible that certain genetic or molecular mechanisms allow for the retention of characteristics from earlier life stages. This could involve shared gene regulatory networks or conserved developmental pathways between different life stages. In other words, some basic biological processes might be common across multiple life stages, allowing for continuity and flexibility in an organism's development.\n\nThese factors could potentially allow for similar transformations to occur in other animals, such as insects that undergo a complete metamorphosis from aquatic larvae to terrestrial adults with wings."
        },
        {
            "role": "teacher",
            "query": "You've brought up some fascinating ideas about preadaptation, developmental plasticity, and shared genetic or molecular mechanisms. Let's dig deeper into these concepts.\n\nWhat do you think might be the advantages of having preadaptations that can be repurposed for new functions? Would this allow organisms to respond more quickly to changes in their environment, or is it more about conserving energy by leveraging existing traits?\n\nAnd regarding developmental plasticity, are there any examples where you've seen an organism's development being strongly influenced by its environment, leading to significant variations in phenotype? How might these environmental influences shape the course of an organism's life history?"
        },
        {
            "role": "student",
            "query": "I think one major advantage of having preadaptations that can be repurposed for new functions is that it would allow organisms to respond more quickly and efficiently to changes in their environment. For example, if amphibian larvae had to suddenly adapt to living on land without the benefit of preexisting traits like limbs and lungs, they might struggle to survive.\n\nHowever, because they already possess these characteristics, albeit in a different form, they can leverage them for new purposes, such as using their gills to extract oxygen from air or modifying their tail fins into limbs. This ability to repurpose existing traits would enable amphibians to adapt more rapidly to changing conditions and potentially stay ahead of predators or competitors.\n\nRegarding developmental plasticity, I'm reminded of the monarch butterfly's incredible ability to migrate thousands of miles each year. Research has shown that these butterflies are able to change their body shape and physiology in response to environmental cues, such as temperature and photoperiod, which allows them to prepare for migration and survive during the journey.\n\nSimilarly, some species of fish have been observed to undergo remarkable transformations in response to changes in water quality or other environmental stressors. For instance, some fish can develop larger gills or modify their digestive systems to cope with pollution or reduced oxygen levels.\n\nIn these cases, developmental plasticity appears to play a key role in allowing the organisms to adapt and survive in response to changing conditions. By influencing an organism's development and shaping its phenotype, environmental factors can have profound effects on its life history and ultimately determine its success or failure in a given environment."
        },
        {
            "role": "teacher",
            "query": "What role do you think the concept of developmental plasticity plays in the overall process of adaptation, particularly when it comes to the transition from one life stage to another? Is it more about responding to specific environmental cues, or is there something more fundamental at play that allows organisms to adapt and change their development in response to changing conditions?"
        },
        {
            "role": "student",
            "query": "I think developmental plasticity plays a crucial role in the overall process of adaptation, particularly when it comes to the transition from one life stage to another. It's not just about responding to specific environmental cues, but rather, it's an inherent ability within organisms themselves that allows them to adapt and change their development in response to changing conditions.\n\nThis concept seems to be closely tied to the idea of preadaptation, where existing traits can be repurposed or modified to serve new functions. When we consider the examples you mentioned earlier, such as the monarch butterfly's migration patterns and certain fish species' ability to adapt to water quality changes, it's clear that developmental plasticity is a key driver of this process.\n\nI'd argue that developmental plasticity allows organisms to tap into their existing genetic or molecular mechanisms, which can then be reconfigured in response to environmental cues. This doesn't necessarily mean that they're starting from scratch each time; rather, they're leveraging their pre-existing traits and modifying them to suit new purposes.\n\nIn a way, you could say that developmental plasticity is an example of \"evolutionary anticipation,\" where organisms are able to anticipate and prepare for potential environmental changes by modifying their development in response. This ability to adapt and change would allow organisms to stay ahead of changing conditions and ultimately increase their chances of survival.\n\nRegarding the concept of \"something more fundamental\" at play, I'd propose that this might involve shared genetic or molecular mechanisms across multiple life stages, which can facilitate continuity and flexibility in an organism's development. These underlying processes could provide a kind of \"developmental scaffolding\" that allows organisms to adapt and change their development in response to changing conditions.\n\nThis idea raises interesting questions about the nature of developmental plasticity itself: Is it a unique property of certain organisms, or is it a more universal process that underlies many different forms of life?"
        },
        {
            "role": "teacher",
            "query": "You've brought up some fascinating ideas about developmental plasticity and its role in adaptation. Let's explore this concept further.\n\nDo you think the ability to tap into existing genetic or molecular mechanisms during development is unique to certain species, such as amphibians and monarch butterflies, or is it a more fundamental property of life that could be applied across many different domains? And what might be the implications if developmental plasticity were found to be a universal process in biology?\n\nHow do you think this concept relates to our understanding of evolution itself? For example, if organisms are able to adapt and change their development in response to changing conditions through mechanisms like preadaptation and developmental plasticity, does that suggest that evolution might not always be a gradual process? Could there be instances where organisms undergo rapid or dramatic changes in response to environmental pressures?\n\nConsider the idea of \"developmental scaffolding\" as you mentioned earlier. What would be some key characteristics of this scaffolding system that would allow it to facilitate continuity and flexibility in an organism's development across multiple life stages?"
        }
    ]
]





class ChatHistory:
    def __init__(self, a):
        self.a = a
    def add_student(self, b):
        self.b = b


class ChatHistoryTree:

    def __init__(self, seed_question):
        self.root = ChatHistory("Groot")  # The root ChatHistory object
        self.root.add_student(seed_question)  # Seed question added to root
        self.tree = {}  # A dictionary of node paths and ChatHistory
        self.tree[(1,)] = self.root

    def add_child(self, parent_path, chat_history):
        """Add a child to the tree under the parent node specified by parent_path."""
        child_path = parent_path (5,) # Path for the new child
        self.tree[child_path] = chat_history  # Store child in nodes dictionary

    def get_node(self, path):
        """Retrieve a ChatHistory node by its path."""
        return self.tree[path]

j = ChatHistoryTree("Studenting")
kk = ChatHistory("I'm kk")
j.add_child((0,), kk)
x = j.get_node((5,))
print(x.a)








=====
extrac_tools.py
=====
import requests
from bs4 import BeautifulSoup
import re

def extract_knowledge(link):
    response = requests.get(link)
    soup = BeautifulSoup(response.text, 'html.parser')

    text_list = []
    for paragraph in soup.find_all('p'):
        text = paragraph.getText().strip()
        text = re.sub(r'\[.*?]', '', text)
        if text: text_list.append(text)

    knowledge = '\n'.join(text_list)
    return knowledge

# Obtain list of urls from url list
with open('url_list.txt') as f:
    url_list = [line.replace('\n', '') for line in f]

# Retrieve and save knowledge from urls
for link in url_list:
    filename = re.search(r'/([^/]+)$', link).group(1)
    information = extract_knowledge(link)
    with open('../knowledge/'+filename+'.txt', 'w') as f:
        f.write(information)

=====

# h = ChatHistory()
# h.add_student("Why is the sky blue?")
# h.add_teacher("Could it be the angle of the sun?")
# h.add_student("Perhaps the blue light is spread by the atmosphere giving it a blue tint.")
# h.add_teacher("Exactly! That's also the reason the sky turns orange during sunrise and sunset.")
#
# j = ChatHistory()
# j.add_student("What is love?")
# j.add_teacher("What do you think it is?")
# j.add_student("Perhaps when we are willing to do anything for another person?")
# j.add_teacher("Of course!.")
#
#
# a = str(h)
# b = h.get_history()
# c = j.get_history()
# d = [b,c]
# json.dump(d, args.o, indent=4)
# judge_response = judge("The sky", "The sky is blue", h)
# print("I'm judge response: " + str(judge_response))
# x = [judge_response]
# for result in x:
#     with open('datasets/' + 'results_val.txt', 'w') as f:
#         f.write(str(result))

# file_path = 'datasets/' + 'eval.json'
# with open(file_path, 'r') as file:
#     data = json.load(file)

# print(type(data), type(data[0]))
# print(data[0])
# print(type(data[0][0]))
# print(data[0][0])
# histories_list = [ChatHistory.from_history(exchange) for exchange in data]
# out = [h.get_history() for h in histories_list]
# json.dump(out, args.o, indent=4)


# h = ChatHistory()
# h.add_text_chunk("The sky is blue because of things.")
# h.add_student("Why is the sky blue?")
# h.add_teacher("Could it be the angle of the sun?")
# h.add_student("Perhaps the blue light is spread by the atmosphere giving it a blue tint.")
# h.add_teacher("Exactly! That's also the reason the sky turns orange during sunrise and sunset.")

j = [{'role': 'text_chunk', 'query': 'The sky is blue because of things.'},
     {'role': 'student', 'query': 'Why is the sky blue?'},
     {'role': 'teacher', 'query': 'Could it be the angle of the sun?'},
     {'role': 'student', 'query': 'Perhaps the blue light is spread by the atmosphere giving it a blue tint.'},
     {'role': 'teacher', 'query': "Exactly! That's also the reason the sky turns orange during sunrise and sunset."}]
k = [{'role': 'text_chunk', 'query': 'Pineapple.'}, {'role': 'student', 'query': 'Why fruit?'},
     {'role': 'teacher', 'query': 'Delicious, no?'}, {'role': 'student', 'query': 'True!'},
     {'role': 'teacher', 'query': "Amen."}]

l = ChatHistory.from_history(j)
m = ChatHistory.from_history(k)
n = [o.get_history() for o in [l, m]]
json.dump(n, args.o, indent=4)


Gemini

import google.generativeai as genai
    import os
    # Access the API key from the environment variable
    api_key = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')

    # Configure the API key
    genai.configure(api_key=api_key)

    model_name = "gemini-1.5-flash"  # Choose the appropriate model name (check Gemini documentation)
    model = genai.GenerativeModel(model_name)
    # response = model.generate_content("The opposite of hot is")
    # print(response.text)

    # prompt = f"Topic: {seed}\n{history}"  # Combine topic and history into a single prompt

    messages = [{'role':'system', 'parts': ['You are a chicken which quacks like duck.']},
                [{'role':'user', 'parts': ['How are you my sweet chicken? Did you lay eggs?']}]]

    prompt = 'How are you my sweet chicken? Did you lay eggs?'
    chicken_role = 'You are a chicken which quacks like duck.'
    response = model.generate_content(f"You will play this role: {chicken_role} \n"
                                      f"As you play this role a person says {prompt}")
    #                                   , role_play=[
    #     {"role": "system", "content": chicken_role}  # Teacher role play
    # ])
    print(response.text)

    # >>> messages = [{'role':'user', 'parts': ['hello']}]
    # >>> response = model. generate_content(messages) # "Hello, how can I help"
    # >>> messages. append(response. candidates[0].content)
    # >>> messages. append({'role':'user', 'parts': ['How does quantum physics work?']})
    # >>> response = model. generate_content(messages


    #     prompt, role_play=[
    #     {"role": "system", "content": student_role},  # Optional role play for student
    # ])

    student_response = response.text

    print("hello")




=================================================================

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

=================================================================

"""Query functions by Gemini"""
api_key = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
genai.configure(api_key=api_key)
model_name = "learnlm-1.5-pro-experimental"
model = genai.GenerativeModel(model_name)

def prompter(role:str, content:str) -> str:
    message = (f"You shall play the role given below. Role: \n {role} \n"
               f"The person you are speaking to gives you the following content."
               f"Content: \n {content} \n")
    return message

def gemini_gen_soc_question(content):
    full_prompt = prompter(query_role, content)
    response = model.generate_content(full_prompt)
    questions = response.text
    return questions


def gemini_gen_seed(content):
    full_prompt = prompter(seed_role, content)
    response = model.generate_content(full_prompt)
    seed = response.text
    return seed

def gemini_gen_student_response(text_chunk, seed_question, history_str):
    content = ("Overall topic: " + text_chunk +
               "\n Seed question: " + seed_question +
               "\n Conversation History: " + history_str)
    full_prompt = prompter(student_role, content)
    response = model.generate_content(full_prompt)
    student_response = response.text
    return student_response

def gemini_gen_teacher_response(content):
    full_prompt = prompter(teacher_role, content)
    response = model.generate_content(full_prompt)
    teacher_response = response.text
    return teacher_response

def gemini_judge(seed:str, text_chunk:str, history:str) -> int:
    content = ("Overall topic: " + text_chunk +
               "\n Seed question: " + seed +
               "\n Conversation History: " + history)
    full_prompt = prompter(judge_role, content)
    response = model.generate_content(full_prompt)
    judge_response = response.text
    return judge_response

=================================================================
"""Query functions by Open AI"""

def openai_gen_soc_question(content):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": query_role},
            {"role": "user", "content": content}
        ]
    )
    questions = response.choices[0].message.content
    return questions

def openai_gen_seed(content):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": seed_role},
                  {"role": "user", "content": content}])
    seed = response.choices[0].message.content
    return seed

def openai_gen_student_response(text_chunk, seed_question, history_str):
    client = OpenAI()
    content = ("Overall topic: " + text_chunk +
               "\n Seed question: " + seed_question +
               "\n Conversation History: " + history_str)
    # content = "Topic: " + seed + "\n Conversation History: " + history # if history else None
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": student_role},
                  {"role": "user", "content": content}])
    student_response = response.choices[0].message.content
    return student_response

def openai_gen_teacher_response(content):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": teacher_role},
                  {"role": "user", "content": content}])
    teacher_response = response.choices[0].message.content
    return teacher_response

def openai_gen_judge(text_chunk, seed, history):
    client = OpenAI()
    content = ("Overall topic: " + text_chunk +
               "\n Seed question: " + seed +
               "\n Conversation History: " + history)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": judge_role},
                  {"role": "user", "content": content}])
    judge_response = response.choices[0].message.content
    return judge_response

=================================================================

