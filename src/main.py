import extract_tools as et
import query_tools as qt


# Define url and save knowledge in text file
url = "https://en.wikipedia.org/wiki/Fish"
fish_text = et.extract_knowledge(url)
with open('../knowledge/fish_text.txt', 'w') as f:
    f.write(fish_text)


# Load knowledge from text file
with open('../knowledge/fish_text.txt') as f:
    lines = [line for line in f]

# Generate socratic questions from a line within knowledge
n = 5
socratic_questions = qt.generate_socratic_questions(lines[n])

# Save socratic questions in text file
with open("../queries/socratic_questions.txt", "w", encoding="utf-8") as output_file:
    output_file.write(socratic_questions)