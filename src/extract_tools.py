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