import copy
class Car:
    def __init__(self, a):
        self.a = a
    def add_student(self, b):
        self.b = b

jack = Car("lifan")
print("jack ", jack.a)
jork = copy.deepcopy(jack)
jork.a = "abay"
print("jack ", jack.a)
print("jork ", jork.a)


def generate_exchanges(seed, history:ChatHistory, iter:int, depth:int) -> list:
    """Generate iter Socratic responses by the teacher"""
    if depth == 0: return [0];
    student_query = student(seed, history)
    history.add_student(student_query) # Student response

    histories = []

    for _ in range(iter): # Multiple teacher responses
        new_history = copy.deepcopy(history)
        teacher_query = teacher(new_history)

        new_history.add_teacher(teacher_query)
        item_histories = generate_exchanges(seed, new_history, iter, depth - 1)
        histories.append(new_history)
        histories.append(item_histories)

    return histories


    # Print the result
    def print_tree(exchanges, level=0, history_number=1):
        """Helper function to print the tree structure with tabs and numbering"""
        for index, exchange in enumerate(exchanges, start=history_number):
            print(" " * (level * 4) + f"History {index}:")  # Indentation with tabs for clarity
            print(" " * (level * 4 + 2) + str(exchange['history']))  # Print the current history

            if exchange['children']:  # If there are child histories, recurse
                print(" " * (level * 4 + 2) + "Children:")
                print_tree(exchange['children'], level + 1, history_number=index + 1)


    # Now, you can run this function after generating the exchanges to display the tree clearly.

    # Print the conversation tree
    print_tree(exchanges)