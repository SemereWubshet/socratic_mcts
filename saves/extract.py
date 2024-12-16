import wikipedia
import random

def download_wiki_pages():
    # Set the language of the Wikipedia pages (default is English)
    wikipedia.set_lang("en")

    # List of pages
    pages_list = ['Art' , 'Technology', 'History', 'Literature',
                  'Philosophy', 'Geography', 'Economics', 'Sports',
                  'Health', 'Psychology', 'Education', 'Religion',
                  'Mathematics', 'Astronomy', 'Culture', 'Engineering']

    # Save all pages in one file
    with open('../knowledge/old_knowledge.txt', 'w') as f:
        for page_name in pages_list:
            page = wikipedia.page(page_name)
            content = page.content
            f.write(f"\n\nTitle: {page_name}\n")
            f.write(f"{content}\n")
    return None

def save_random_wiki_pages(filename, num_pages=2):

  with open(filename, 'w') as f:
    for _ in range(num_pages):
        random_title = random.choice(wikipedia.random(pages=num_pages))
        try:
            page = wikipedia.page(random_title)
            content = page.content
            f.write(f"Title: {page.title}\n")
            f.write(f"{content}\n")
        except wikipedia.exceptions.DisambiguationError as e:
            # Choose the first disambiguation page
            first_page_title = e.options[0]
            first_page = wikipedia.page(first_page_title)
            f.write(f"Random Page (disambiguation): {random_title}\n (Selected: {first_page_title})\n\n")
            f.write(first_page.content)
            f.write("\n\n")

if __name__ == "__main__":
  save_random_wiki_pages("../knowledge/new_knowledge.txt", 15)