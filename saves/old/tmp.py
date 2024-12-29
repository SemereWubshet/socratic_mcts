from saves.old.conversation_generator import gen_seed_question

if __name__ == "__main__":
    text_chunk = ("The fire triangle or combustion triangle is a simple model for understanding the necessary "
                  "ingredients for most fires [1]. The triangle illustrates the three elements a fire needs to "
                  "ignite: heat, fuel, and an oxidizing agent (usually oxygen).[2] A fire naturally occurs when the "
                  "elements are present and combined in the right mixture.[3] A fire can be prevented or extinguished "
                  "by removing any one of the elements in the fire triangle. For example, covering a fire with a fire "
                  "blanket blocks oxygen and can extinguish a fire. In large fires where firefighters are called in, "
                  "decreasing the amount of oxygen is not usually an option because there is no effective way to make "
                  "that happen in an extended area.[4]")

    seeds = [gen_seed_question(text_chunk) for i in range(10)]

    print("\n".join(seeds))

    # # history = generate_exchange(text_chunk)
    #
    # seed_question = "How can I stop fire?"
    #
    # history = ChatHistory()
    # history.add_text_chunk(text_chunk)
    # history.add_student_type(7)
    # history.add_student(seed_question)
    #
    # teacher_query = ("That are many ways one can stop a fire. Are you thinking about a specific fire source, types of "
    #                  "extinguishers, or to the basic elements of a fire?")
    # history.add_teacher(teacher_query)
    #
    # student_query = student(text_chunk, seed_question, history)
    # history.add_student(student_query)
    #
    # teacher_query = ("The fire triangle says that fire in a combination of fuel, heat and oxygen. Removing any of "
    #                  "these elements will stop the fire. If we throw water in a fire, it stops. What element "
    #                  "did it remove?")
    # history.add_teacher(teacher_query)
    #
    # student_query = student(text_chunk, seed_question, history)
    # history.add_student(student_query)
