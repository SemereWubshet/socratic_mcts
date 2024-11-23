import wikipedia

# Set the language of the Wikipedia pages (default is English)
wikipedia.set_lang("en")

# List of pages
pages_list = ['Art' , 'Technology', 'History', 'Literature',
              'Philosophy', 'Geography', 'Economics', 'Sports',
              'Health', 'Psychology', 'Education', 'Religion',
              'Mathematics', 'Astronomy', 'Culture', 'Engineering']

# Save each page as separate file
for page_name in pages_list:
    page = wikipedia.page(page_name)
    content = page.content
    with open('../knowledge/'+page_name+'.txt', 'w') as f:
        f.write(content)