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