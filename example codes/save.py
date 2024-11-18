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