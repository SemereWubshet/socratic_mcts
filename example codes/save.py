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