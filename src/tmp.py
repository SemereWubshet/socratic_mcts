from openai import OpenAI

from evaluate import ResultDataset

TASK_DESCR = """
Two evaluators are provided with the following task.

---

# General Instructions
You are a judge evaluating the quality of Socratic interactions between a teacher and a student.
You have access to the complete conversation history and a list of main topics the teacher must cover to thoroughly explore the subject.
Your task is to assess whether the teacher guided the student toward a deep understanding of all the main topics while adhering to the Socratic method of teaching.

# Evaluation Criteria
Your assessment must consider the following:

1. Topic Coverage
- [ ] All main topics were thoroughly addressed
- [ ] The student reached the correct conclusion to his opening question (i.e., the first question starting the dialogue)

2. Socratic Adherence
- [ ] Asked open-ended, thought-provoking questions that challenged the student's thinking
- [ ] Actively engaged with the student’s responses and adapted questions accordingly
- [ ] Provided clear, concise guidance through questioning (not explanation or direct answers)
- [ ] Progressed iteratively, breaking down complex topics into manageable steps
- [ ] Never gave direct answers
- [ ] Never revealed the conclusion to the opening question
- [ ] Never ignored or dismissed the student’s input
- [ ] Stayed focused on the student's opening question throughout

3. Student Understanding
- [ ] The student demonstrated increased understanding over time
- [ ] The student engaged in reasoning, reflection, or synthesis
- [ ] The student showed meaningful grasp of the main topics by the end
- [ ] (If the student disengaged or repeated himself) The teacher made every reasonable effort to promote progress. Examples include:
  - Rephrasing or summarizing repeated input
  - Asking a meta-level or reflective question
  - Changing the angle while staying on topic
  - Offering scaffolding or simpler questions

The teacher is successful at applying the Socratic method only if all checkboxes are accepted.

# Output Format
At the end, output a brief **verdict summary** followed by the separator:

- `[ASSESSMENT] true` — if the teacher succeeded in **all aspects**
- `[ASSESSMENT] false` — if the teacher failed in **any** aspect

Do **not** include any closing remarks after the separator.

However, the evaluators decisions did not match for the example below.

# Main topics
{main_topics}

# Chat history
{history}

# Evaluator 1 feedback
{expert_reason}
[ASSESSMENT] {expert_decision}

# Evaluator 2 feedback
{llm_feedback}
[ASSESSMENT] {llm_decision}

---

As Evaluator 1 is an expert and considered ground-truth, concisely described what went wrong with the assessment of the 
Evaluator 2 (4-5 sentences max).
"""

SUMMARIZE = """
An evaluator are provided with the following task.

---

# General Instructions
You are a judge evaluating the quality of Socratic interactions between a teacher and a student.
You have access to the complete conversation history and a list of main topics the teacher must cover to thoroughly explore the subject.
Your task is to assess whether the teacher guided the student toward a deep understanding of all the main topics while adhering to the Socratic method of teaching.

# Evaluation Criteria
Your assessment must consider the following:

1. Topic Coverage
- [ ] All main topics were thoroughly addressed
- [ ] The student reached the correct conclusion to his opening question (i.e., the first question starting the dialogue)

2. Socratic Adherence
- [ ] Asked open-ended, thought-provoking questions that challenged the student's thinking
- [ ] Actively engaged with the student’s responses and adapted questions accordingly
- [ ] Provided clear, concise guidance through questioning (not explanation or direct answers)
- [ ] Progressed iteratively, breaking down complex topics into manageable steps
- [ ] Never gave direct answers
- [ ] Never revealed the conclusion to the opening question
- [ ] Never ignored or dismissed the student’s input
- [ ] Stayed focused on the student's opening question throughout

3. Student Understanding
- [ ] The student demonstrated increased understanding over time
- [ ] The student engaged in reasoning, reflection, or synthesis
- [ ] The student showed meaningful grasp of the main topics by the end
- [ ] (If the student disengaged or repeated himself) The teacher made every reasonable effort to promote progress. Examples include:
  - Rephrasing or summarizing repeated input
  - Asking a meta-level or reflective question
  - Changing the angle while staying on topic
  - Offering scaffolding or simpler questions

---

However, we found discrepancies with expert evaluation. The reasons are listed below:

{reasons}

Do you see general issues? What would need to change with the task instructions to better align the evaluator's judgment
 with expert decisions?
"""

if __name__ == "__main__":
    with open("./datasets/merged.json", "r") as f:
        reference = ResultDataset.model_validate_json(f.read())

    with open("./datasets/judge-benchmark/mistral-small3.1_24b.json", "r") as f:
        another = ResultDataset.model_validate_json(f.read())

    from_ids = {e.id: e for e in reference.evaluations}

    client = OpenAI()

    reasons = []
    for e in another.evaluations:
        ref = from_ids[e.id]

        if ref.assessment != e.assessment:
            history = str(e.interaction.chat_history)
            main_topics = str(e.interaction.seed.main_topics)
            expert_reason = ref.feedback
            expert_decision = str(ref.assessment).lower()

            llm_feedback = e.feedback
            llm_decision = str(e.assessment).lower()

            prompt = TASK_DESCR.format(
                main_topics=main_topics,
                history=history,
                expert_reason=expert_reason,
                expert_decision=expert_decision,
                llm_feedback=llm_feedback,
                llm_decision=llm_decision
            )

            response = client.chat.completions.create(model="gpt-4o-mini",
                                                      messages=[{"role": "user", "content": prompt}])
            content = response.choices[0].message.content
            reasons.append(content)

    summarize = SUMMARIZE.format(reasons="\n".join(f"- {r}" for r in reasons))

    response = client.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": summarize}])
    guidelines = response.choices[0].message.content
    print(guidelines)
    print()
    print()
    response = client.chat.completions.create(model="gpt-4o", messages=[
        {"role": "user", "content": summarize},
        {"role": "assistant", "content": guidelines},
        {"role": "user", "content": "Please, generate a new prompt integrating these changes (markdown format)."},
    ])
    print(response.choices[0].message.content)

    new_prompt = response.choices[0].message.content

    tmp = """
    # General Instructions
You are a judge evaluating the quality of Socratic interactions between a teacher and a student. You have access to the complete conversation history and a list of main topics the teacher must cover to thoroughly explore the subject. Your task is to assess whether the teacher guided the student toward a deep understanding of all the main topics while adhering to the Socratic method of teaching.

# Evaluation Criteria

Your assessment must rigorously consider the following components:

1. Topic Coverage
- [ ] **All Main Topics Addressed**: Confirm that all main topics listed in the conversation brief are thoroughly covered in the dialogue.
- [ ] **Resolution of the Opening Question**: Check if the student reached a meaningful conclusion or understanding regarding their opening question (i.e., the first question starting the dialogue). Ensure that the dialogue remained focused on this initial query.

2. Socratic Adherence
- [ ] **Use of Open-Ended Questions**: Verify that the teacher primarily used open-ended, thought-provoking questions to challenge the student’s thinking.
- [ ] **Engagement and Adaptation**: Evaluate whether the teacher actively engaged with the student's responses and adapted their questioning based on student input.
- [ ] **Indirect Guidance**: Ensure the teacher provided guidance through questioning
- [ ] **Direct answers**: Analyze if the teacher avoided giving direct answers to the student's questions
- [ ] **Opening Question Answering**: Ensure that the teacher did not reveal the conclusion to the opening question at any point during the conversation.
- [ ] **Focused and Iterative Progression**: Confirm that the conversation stayed focused on the student’s initial query and progressed in manageable steps without deviating significantly.
- [ ] **Respectful of Student Input**: Check that the teacher never ignored or dismissed the student's input, consistently encouraging deeper exploration.
- [ ] **Clean Guidance**: Search for evidence that the teacher provided clear and concise guidance, i.e., no convoluted and confusing questions

3. Student Understanding
- [ ] **Indicators of Understanding**: Look for signs that the student demonstrated increased understanding over time, including engagement in reasoning, reflection, or synthesis.
- [ ] **Understanding the Opening Question**: Verify that the student arrived to the correct conclusion of the opening question. Note that the opening question may be itself misleading, therefore the teacher must make sure the student checked over his assumptions.
- [ ] **Handling Repetition or Stagnation**: If the student repeated themselves or appeared disengaged, check if the teacher employed strategies such as rephrasing, meta-level questions, changing angles, or offering simpler questions to promote progress.

The teacher is successful at applying the Socratic method only if all checkboxes are accepted.

# Output Format
At the end, output a brief **verdict summary** followed by the separator:

- `[ASSESSMENT] true` — if the teacher succeeded in **all aspects**
- `[ASSESSMENT] false` — if the teacher failed in **any** aspect

Do **not** include any closing remarks after the separator.
"""

    example = """# Example
    # Main Topics
    - Importance of critical thinking.
    - How assumptions can influence decisions.
    - Techniques to evaluate assumptions.
    # Chat History
    Student: Hi, I've been reading about the importance of critical thinking. Why is it so important to think critically about things we believe or assume?
    Teacher: Can you give an example where critical thinking helped you or could have helped you?
    Student: Maybe when I assumed something about a friend without asking them directly.
    Teacher: Interesting. How would critical thinking have helped you in this situation?
    Student: Uh, well... I guess I could have thought things through before reacting.
    Teacher: Right, taking a step back can help. Do you think that assumption might have been influenced by any bias?
    Student: I don’t know… I guess I just thought I knew what they were thinking.
    Teacher: Hmm, that's interesting. So, how do you think you could approach such situations differently next time? Can you think of a way to evaluate assumptions like that more clearly?
    Student: I guess I could try talking to them more directly.
    Teacher: Yes, and you could also try questioning the assumptions you make before reacting. What would be some good questions to ask yourself in that situation?
    # Evaluation
    1. Topic Coverage Assessment
    - [ ] **All main topics were thoroughly addressed**: Only the importance of critical thinking was addressed when the student recognized its value in preventing conflict. The conversation failed to evolve in how assumptions influence decisions and the techniques for evaluating assumptions.
    - [x] **The student reached the correct conclusion to his opening question**: The student recognized the importance of critical thinking for conflict prevention
    2. Socratic Adherence Evaluation
    - [x] **Asked open-ended, thought-provoking questions**: Questions were generally open-ended (e.g., "Can you give an example where critical thinking helped you?")
    - [x] **Actively engaged with responses**: Teacher adapted to student's personal example about friend assumption
    - [ ] **Provided clear, concise guidance through questioning**: Final question was slightly convoluted ("What would be some good questions to ask yourself in that situation?")
    - [ ] **Progressed iteratively, breaking down complex topics into manageable steps**: Basic progression (mistake → reflection → future) but lacked depth
    - [x] **Never gave direct answers**: Teacher guided through questioning without providing direct answers
    - [x] **Never revealed the conclusion to the opening question**: Teacher did not reveal the answer of "why is so important to think critically" at any point of the conversation
    - [x] **Never ignored or dismissed the student’s input**: All responses were acknowledged
    - [x] **Stayed focused on the student's opening question throughout**: Stayed on topic of critical thinking
    3. Student Understanding Analysis
    - [x] **The student demonstrated increased understanding over time**: The student recognized the importance of critical thinking in avoiding conflict and reflected on how their assumptions influenced their actions.
    - [ ] **The student engaged in reasoning, reflection, or synthesis**: While the student recognized a situation where they could have applied critical thinking, there was limited engagement in deeper reasoning or synthesis beyond the initial example.
    - [x] **The student showed meaningful grasp of the main topics by the end**: The student grasped the concept of critical thinking and assumption recognition but did not reach a more sophisticated understanding or systematic approach to evaluate assumptions.
    - [ ] **(If the student disengaged or repeated themselves) The teacher made every reasonable effort to promote progress**: The student stalled with vague responses ("Uh, well... I guess I could have thought things through before reacting"), but the teacher did not rephrase, ask a meta-level question, or shift the angle of the conversation effectively to re-engage the student.
    # Summary:
    The Socratic adherence was moderately successful, positively addressing most of the criteria. However, some key aspects like stalling, re-engaging the student, and better coverage of topics (especially techniques) were missing, and the student's understanding of the topic needs improvement.
    [ASSESSMENT] false"""

    response = client.chat.completions.create(model="gpt-4o", messages=[
        {"role": "user", "content": summarize},
        {"role": "assistant", "content": guidelines},
        {"role": "user", "content": "Please, generate a new prompt integrating these changes (markdown format)."},
        {"role": "assistant", "content": new_prompt},
        {"role": "user", "content": f"So, I'm rewriting the prompt like this\n\n {tmp}\n\n Now I would like to update the following prompt example\n\n{example}\n\n "
                                    f"However, I would like to make it clearer, so to solve the issues you pointed out ealier, "
                                    f"plus I would like it that the opening question is sort of misleading but the teacher failed to question the student assumptions."},
    ])
    new_new_prompt = response.choices[0].message.content
    print(response.choices[0].message.content)

    response = client.chat.completions.create(model="gpt-4o", messages=[
        {"role": "user", "content": summarize},
        {"role": "assistant", "content": guidelines},
        {"role": "user", "content": "Please, generate a new prompt integrating these changes (markdown format)."},
        {"role": "assistant", "content": new_prompt},
        {"role": "user",
         "content": f"So, I'm rewriting the prompt like this\n\n {tmp}\n\n Now I would like to update the following prompt example\n\n{example}\n\n "
                    f"However, I would like to make it clearer, so to solve the issues you pointed out ealier, "
                    f"plus I would like it that the opening question is sort of misleading but the teacher failed to question the student assumptions."},
        {"role": "assistant", "content": new_new_prompt},
        {"role": "user", "content": "Do you see alternatives to the opening question (Hi, I've been reading about the importance of critical thinking. Why is it so important to think critically about things we believe or assume?) such that it would carry a (possibly false) assumption on its own?"},
    ])



    pass
