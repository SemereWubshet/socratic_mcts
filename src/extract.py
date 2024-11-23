import wikipedia

# Set the language of the Wikipedia pages (default is English)
wikipedia.set_lang("en")

# List of pages
pages_list = ['Art' , 'Technology', 'History', 'Literature',
              'Philosophy', 'Geography', 'Economics', 'Sports',
              'Health', 'Psychology', 'Education', 'Religion',
              'Mathematics', 'Astronomy', 'Culture', 'Engineering']

# Save all pages in one file
with open('../knowledge/diverse_knowledge.txt', 'w') as f:
    for page_name in pages_list:
        page = wikipedia.page(page_name)
        content = page.content
        f.write(f"\n\nTitle: {page_name}\n")
        f.write(f"{content}\n")