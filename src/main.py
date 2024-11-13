import extract_tools as et
import query_tools as qt
import re


# Obtain list of urls from url list
with open('url_list.txt') as f:
    url_list = [line.replace('\n', '') for line in f]


# x = et.extract_knowledge(url_list[0])
# print(x)

# Retrieve and save knowledge from urls
for link in url_list:
    filename = re.search(r'/([^/]+)$', link).group(1)
    information = et.extract_knowledge(link)
    with open('../knowledge/'+filename+'.txt', 'w') as f:
        f.write(information)


# # Load knowledge from text file
# with open('../knowledge/fish_text.txt') as f:
#     lines = [line for line in f]
#
# # Generate socratic questions from a line within knowledge
# n = 5
# socratic_questions = qt.openai_gen_soc_questions(lines[n])
#
# # Save socratic questions in text file
# with open("../queries/socratic_questions.txt", "w", encoding="utf-8") as output_file:
#     output_file.write(socratic_questions)