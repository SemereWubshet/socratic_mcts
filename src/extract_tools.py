import requests
import re
from bs4 import BeautifulSoup

def extract_knowledge(link):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    text_list = []
    for paragraph in soup.find_all('p'):
        text = paragraph.getText().strip()
        text = re.sub(r'\[.*?]', '', text)
        if text: text_list.append(text)

    knowledge = '\n'.join(text_list)
    return knowledge

url = "https://en.wikipedia.org/wiki/Fish"
