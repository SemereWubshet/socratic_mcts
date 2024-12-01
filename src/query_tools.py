import random
import ollama
import pathlib
import os
import google.generativeai as genai
from openai import OpenAI

# with open('templates/query_generator.txt', 'r') as f: # Make a keypoints.txt role
#     keypoints_role = f.read()

keypoints_role = "Given the piece of text below, extract major keypoints necessary for comprehensive understanding."

with open('templates/query_generator.txt', 'r') as f:
    query_role = f.read()

with open('templates/judge.txt', 'r') as f:
    judge_role = f.read()

with open('templates/seed.txt', 'r') as f:
    seed_role = f.read()

with open('templates/student.txt', 'r') as f:
    student_role = f.read()

with open('templates/teacher.txt', 'r') as f:
    teacher_role = f.read()

"""Interaction types"""

INTERACTION_TYPES = (
    "Demand deeper clarification about one of the major points on the topic.",
    "Request further explanations that go beyond the original text.",
    "Make misleading claims due to misunderstanding on one or more of the topics.",
    "Act confused about one of the major points, thus requiring further explanation from the teacher.",
    "Demonstrate inability to connect major points.",
    "Suggest a different understanding of a major point so to lead to a discussion about its validity.",
    "Request examples or applications of a major point in practical, real-world scenarios.",
    "Request the comparison to major points with similar or contrasting concepts.",
    "Pose \"what-if\" questions to explore the implications of the major point in various contexts.",
    "Question the foundational assumptions of the major point, prompting justification or re-explanation.",
    "Request an explanation of the major point in simpler terms or using analogies.",
    "Demonstrate understanding of some basic concepts but struggle to connect them to the broader major point.",
    "Ask questions that are tangentially related to the major point, requiring the teacher to refocus the conversation "
    "while addressing the inquiry.",
    "Ask for a detailed breakdown of a process or concept related to the major point.",
    "Ask if there are any arguments or evidence against the major point, prompting critical evaluation.",
    "Make overly broad statements about the major point, requiring clarification or correction.",
)

"""Query functions by Open AI"""

def openai_gen_soc_question(content):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": query_role},
            {"role": "user", "content": content}
        ]
    )
    questions = response.choices[0].message.content
    return questions

def openai_gen_seed(content):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": seed_role},
                  {"role": "user", "content": content}])
    seed = response.choices[0].message.content
    return seed

def openai_gen_student_response(text_chunk, seed_question, history_str):
    client = OpenAI()
    content = ("Overall topic: " + text_chunk +
               "\n Seed question: " + seed_question +
               "\n Conversation History: " + history_str)
    # content = "Topic: " + seed + "\n Conversation History: " + history # if history else None
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": student_role},
                  {"role": "user", "content": content}])
    student_response = response.choices[0].message.content
    return student_response

def openai_gen_teacher_response(content):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": teacher_role},
                  {"role": "user", "content": content}])
    teacher_response = response.choices[0].message.content
    return teacher_response

def openai_gen_judge(text_chunk, seed, history):
    client = OpenAI()
    content = ("Overall topic: " + text_chunk +
               "\n Seed question: " + seed +
               "\n Conversation History: " + history)
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": judge_role},
                  {"role": "user", "content": content}])
    judge_response = response.choices[0].message.content
    return judge_response


"""Query functions by Ollama"""

def ollama_gen_key_points(content):
    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")
    response = client.chat(model="llama3.1", messages=[{"role": "system", "content": keypoints_role},
                                                       {"role": "user", "content": content}])
    keypoints = response["message"]["content"]
    return keypoints

def ollama_gen_soc_question(content):
    # keypoints = ollama_gen_key_points(content)
    # client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")
    # response = client.chat(model="llama3.1", messages=[{"role": "system", "content": query_role},
    #                                                    {"role": "user", "content": keypoints}])

    base_prompt = pathlib.Path("./templates/seed.txt").read_text(encoding="UTF-8")
    interaction_type = random.choice(INTERACTION_TYPES)
    content = base_prompt.format(context=content, interaction_type=interaction_type)

    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")
    response = client.chat(model="mistral-nemo:12b-instruct-2407-fp16",
                           messages=[{"role": "user", "content": content}])

    question = response["message"]["content"]
    return question

def ollama_gen_seed(content):
    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")
    response = client.chat(model="llama3.1", messages=[{"role": "system", "content": seed_role},
                                                       {"role": "user", "content": content}])
    seed = response["message"]["content"]
    return seed

def ollama_gen_student_response(text_chunk, seed_question, history_str):
    content = ("Overall topic: " + text_chunk +
               "\n Seed question: " + seed_question +
               "\n Conversation History: " + history_str)
    # content = "Topic: " + seed + "\n Conversation History: " + history # if history else None
    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")
    response = client.chat(model="llama3.1", messages=[{"role": "system", "content": student_role},
                                                       {"role": "user", "content": content}])
    student_response = response["message"]["content"]
    return student_response

def ollama_gen_teacher_response(content):
    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")
    # print("\nTeacher role \n", teacher_role)
    # print("\nContent \n", content)
    response = client.chat(model="llama3.1", messages=[{"role": "system", "content": teacher_role},
                                                       {"role": "user", "content": content}])
    teacher_response = response["message"]["content"]
    # print("\nTeacher response \n", teacher_response)
    return teacher_response

def ollama_judge(seed:str, text_chunk:str, history:str) -> int:
    content = ("Overall topic: " + text_chunk +
               "\n Seed question: " + seed +
               "\n Conversation History: " + history)
    client = ollama.Client(host="http://atlas1api.eurecom.fr:8019")
    response = client.chat(model="llama3.1", messages=[{"role": "system", "content": judge_role},
                                                       {"role": "user", "content": content}])
    judge_response = response["message"]["content"]
    return judge_response

"""Query functions by Gemini"""
api_key = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
genai.configure(api_key=api_key)
model_name = "learnlm-1.5-pro-experimental"
model = genai.GenerativeModel(model_name)

def prompter(role:str, content:str) -> str:
    message = (f"You shall play the role given below. Role: \n {role} \n"
               f"The person you are speaking to gives you the following content."
               f"Content: \n {content} \n")
    return message

def gemini_gen_soc_question(content):
    full_prompt = prompter(query_role, content)
    response = model.generate_content(full_prompt)
    questions = response.text
    return questions


def gemini_gen_seed(content):
    full_prompt = prompter(seed_role, content)
    response = model.generate_content(full_prompt)
    seed = response.text
    return seed

def gemini_gen_student_response(text_chunk, seed_question, history_str):
    content = ("Overall topic: " + text_chunk +
               "\n Seed question: " + seed_question +
               "\n Conversation History: " + history_str)
    full_prompt = prompter(student_role, content)
    response = model.generate_content(full_prompt)
    student_response = response.text
    return student_response

def gemini_gen_teacher_response(content):
    full_prompt = prompter(teacher_role, content)
    response = model.generate_content(full_prompt)
    teacher_response = response.text
    return teacher_response

def gemini_judge(seed:str, text_chunk:str, history:str) -> int:
    content = ("Overall topic: " + text_chunk +
               "\n Seed question: " + seed +
               "\n Conversation History: " + history)
    full_prompt = prompter(judge_role, content)
    response = model.generate_content(full_prompt)
    judge_response = response.text
    return judge_response


# # content = "In this chapter we will look at trigonometric series. Previously, we saw that such series expansion occurred naturally in the solution of the heat equation and other boundary value problems. In the last chapter we saw that such functions could be viewed as a basis in an infinite dimensional vector space of functions. Given a function in that space, when will it have a representation as a trigonometric series? For what values of $x$ will it converge? Finding such series is at the heart of Fourier, or spectral, analysis. There are many applications using spectral analysis. At the root of these studies is the belief that many continuous waveforms are comprised of a number of harmonics. Such ideas stretch back to the Pythagorean study of the vibrations of strings, which lead to their view of a world of harmony. This idea was carried further by Johannes Kepler in his harmony of the spheres approach to planetary orbits. In the 1700 's others worked on the superposition theory for vibrating waves on a stretched spring, starting with the wave equation and leading to the superposition of right and left traveling waves. This work was carried out by people such as John Wallis, Brook Taylor and Jean le Rond d'Alembert. In 1742 d'Alembert solved the wave equation $c^{2} \dfrac{\partial^{2} y}{\partial x^{2}}-\dfrac{\partial^{2} y}{\partial t^{2}}=0$, where $y$ is the string height and $c$ is the wave speed. However, his solution led himself and others, like Leonhard Euler and Daniel Bernoulli, to investigate what \"functions\" could be the solutions of this equation. In fact, this lead to a more rigorous approach to the study of analysis by first coming to grips with the concept of a function. For example, in 1749 Euler sought the solution for a plucked string in which case the initial condition $y(x, 0)=h(x)$ has a discontinuous derivative! In 1753 Daniel Bernoulli viewed the solutions as a superposition of simple vibrations, or harmonics. Such superpositions amounted to looking at solutions of the form $y(x, t)=\sum_{k} a_{k} \sin \dfrac{k \pi x}{L} \cos \dfrac{k \pi c t}{L}$, where the string extends over the interval $[0, L]$ with fixed ends at $x=0$ and $x=L$. However, the initial conditions for such superpositions are $y(x, 0)=\sum_{k} a_{k} \sin \dfrac{k \pi x}{L}. \nonumber$ It was determined that many functions could not be represented by a finite number of harmonics, even for the simply plucked string given by an initial condition of the form $y(x, 0)=\left\{\begin{array}{cl} c x, & 0 \leq x \leq L / 2 \ c(L-x), & L / 2 \leq x \leq L \end{array}\right.$ Thus, the solution consists generally of an infinite series of trigonometric functions. Such series expansions were also of importance in Joseph Fourier's solution of the heat equation. The use of such Fourier expansions became an important tool in the solution of linear partial differential equations, such as the wave equation and the heat equation. As seen in the last chapter, using the Method of Separation of Variables, allows higher dimensional problems to be reduced to several one dimensional boundary value problems. However, these studies lead to very important questions, which in turn opened the doors to whole fields of analysis. Some of the problems raised were 1. What functions can be represented as the sum of trigonometric functions? 2. How can a function with discontinuous derivatives be represented by a sum of smooth functions, such as the above sums? 3. Do such infinite sums of trigonometric functions a actually converge to the functions they represents? Sums over sinusoidal functions naturally occur in music and in studying sound waves. A pure note can be represented as $y(t)=A \sin (2 \pi f t)$, where $A$ is the amplitude, $f$ is the frequency in hertz $(\mathrm{Hz})$, and $t$ is time in seconds. The amplitude is related to the volume, or intensity, of the sound. The larger the amplitude, the louder the sound. In Figure 5.1 we show plots of two such tones with $f=2 \mathrm{~Hz}$ in the top plot and $f=5 \mathrm{~Hz}$ in the bottom one. Next, we consider what happens when we add several pure tones. After all, most of the sounds that we hear are in fact a combination of pure tones with different amplitudes and frequencies. In Figure 5.2 we see what happens when we add several sinusoids. Note that as one adds more and more tones with different characteristics, the resulting signal gets more complicated. However, we still have a function of time. In this chapter we will ask, \"Given a function $f(t)$, can we find a set of sinusoidal functions whose sum converges to $f(t)$?\" Looking at the superpositions in Figure 5.2, we see that the sums yield functions that appear to be periodic. This is not to be unexpected. We recall that a periodic function is one in which the function values repeat over the domain of the function. The length of the smallest part of the domain which repeats is called the period. We can define this more precisely. Definition 5.1. A function is said to be periodic with period $T$ if $f(t+T)=f(t)$ for all $t$ and the smallest such positive number $T$ is called the period. For example, we consider the functions used in Figure 5.2. We began with $y(t)=2 \sin (4 \pi t)$. Recall from your first studies of trigonometric functions that one can determine the period by dividing the coefficient of $t$ into $2 \pi$ to get the period. In this case we have $T=\dfrac{2 \pi}{4 \pi}=\dfrac{1}{2} \nonumber$ Looking at the top plot in Figure 5.1 we can verify this result. (You can count the full number of cycles in the graph and divide this into the total time to get a more accurate value of the period.) In general, if $y(t)=A \sin (2 \pi f t)$, the period is found as $T=\dfrac{2 \pi}{2 \pi f}=\dfrac{1}{f} \nonumber$ Of course, this result makes sense, as the unit of frequency, the hertz, is also defined as $s^{-1}$, or cycles per second. Returning to the superpositions in Figure 5.2, we have that $y(t)= \sin (10 \pi t)$ has a period of $0.2 \mathrm{~Hz}$ and $y(t)=\sin (16 \pi t)$ has a period of $0.125 \mathrm{~Hz}$. The two superpositions retain the largest period of the signals added, which is $0.5 \mathrm{~Hz}$. Our goal will be to start with a function and then determine the amplitudes of the simple sinusoids needed to sum to that function. First of all, we will see that this might involve an infinite number of such terms. Thus, we will be studying an infinite series of sinusoidal functions. Secondly, we will find that using just sine functions will not be enough either. This is because we can add sinusoidal functions that do not necessarily peak at the same time. We will consider two signals that originate at different times. This is similar to when your music teacher would make sections of the class sing a song like \"Row, Row, Row your Boat\" starting at slightly different times. We can easily add shifted sine functions. In Figure 5.3 we show the functions $y(t)=2 \sin (4 \pi t)$ and $y(t)=2 \sin (4 \pi t+7 \pi / 8)$ and their sum. Note that this shifted sine function can be written as $y(t)=2 \sin (4 \pi(t+7 / 32))$. Thus, this corresponds to a time shift of $-7 / 8$. So, we should account for shifted sine functions in our general sum. Of course, we would then need to determine the unknown time shift as well as the amplitudes of the sinusoidal functions that make up our signal, $f(t)$. While this is one approach that some researchers use to analyze signals, there is a more common approach. This results from another reworking of the shifted function. Consider the general shifted function $y(t)=A \sin (2 \pi f t+\phi). \nonumber$ Note that $2 \pi f t+\phi$ is called the phase of our sine function and $\phi$ is called the phase shift. We can use our trigonometric identity for the sine of the sum of two angles to obtain $y(t)=A \sin (2 \pi f t+\phi)=A \sin (\phi) \cos (2 \pi f t)+A \cos (\phi) \sin (2 \pi f t). \nonumber$ Defining $a=A \sin (\phi)$ and $b=A \cos (\phi)$, we can rewrite this as $y(t)=a \cos (2 \pi f t)+b \sin (2 \pi f t) \nonumber$ Thus, we see that our signal is a sum of sine and cosine functions with the same frequency and different amplitudes. If we can find $a$ and $b$, then we can easily determine $A$ and $\phi$ : $A=\sqrt{a^{2}+b^{2}} \quad \tan \phi=\dfrac{b}{a}. \nonumber$ We are now in a position to state our goal in this chapter. Goal Given a signal $f(t)$, we would like to determine its frequency content by finding out what combinations of sines and cosines of varying frequencies and amplitudes will sum to the given function. This is called Fourier Analysis."
# content2 = "There are a number of steps that must take place for voluntary movement to occur. Assessment of the surrounding environment and the body’s location in space, followed by determining what action is appropriate, and then initiating that action. We will first focus on the cortical regions involved in planning of voluntary movement. After work, you sit down on the couch to watch one episode of your favorite show. As the end credits appear, you realize it is now time to head to your study space and start working on class. To do this, you need to leave the couch, grab your computer from the table, get your coffee from the kitchen and head to a different room. All of these voluntary movements take a great deal of processing by the brain. You must assess your surrounding environment and your body’s location in it, determine which actions need to be completed, and then actually initiate those actions. In this chapter we will focus on how the planning of voluntary movement occurs. Cortical Anatomy Much of the cortex is actually involved in the planning of voluntary movement. Sensory information, particularly the dorsal stream of the visual and somatosensory pathways, are processed in the posterior parietal lobe where Visual, tactile, and proprioceptive information are integrated. Connections from the posterior parietal lobe are then sent to both the premotor regions and the prefrontal cortex. The prefrontal cortex, which is located in the front of the brain in the frontal lobe, plays an important role in higher level cognitive functions like planning, critical thinking, and understanding the consequences of our behaviors. The premotor area lies just anterior to the primary motor cortex. This region helps plan and organize movement and makes decisions about which actions should be used for a situation. View the primary motor cortex using the BrainFacts.org 3D Brain View the premotor cortex using the BrainFacts.org 3D Brain View the prefrontal cortex using the BrainFacts.org 3D Brain Role of Premotor Area The premotor regions do send some axons directly to lower motor neurons in the spinal cord using the same pathways as the motor cortex (see Execution of Movement chapter). However, the premotor cortex also plays an important role in the planning of movement. Two experimental designs have demonstrated this role. Monkeys were trained on a panel that had one set of lights in a row on top and one set of buttons that could also light up in a row on the bottom. The monkeys would watch for a top row light to turn on. This would indicate that within a few seconds, the button directly below would light up. When the button turned on, the monkeys were supposed to push the button. Therefore, there were two light triggers in the experiment. The first required no motor movement from the monkey but did give the monkey information about where a motor movement would be needed in the near future. The second required the monkey to move to push the button. When brain activity was measured during this study, neurons in the premotor cortex became active when the first light trigger turned on, well before any movement actually took place (Weinrich and Wise, 1928). In another experiment, people were trained to move their fingers in a specific pattern. Cerebral blood flow was then measured when they repeated the finger pattern and when they only imagined repeating the finger pattern. When the movement was only imagined and not actually executed, the premotor regions along with parts of the prefrontal cortex were activated (Roland, et al, 1980). These studies show that the premotor cortex is active prior to the execution of movement, indicating that it plays an important role in the planning of movement. The posterior parietal, prefrontal, and premotor regions, though, also communicate with a subcortical region called the basal ganglia to fully construct the movement plan. The basal ganglia are covered in the next chapter. Key Takeaways • Sensory information is processed in the posterior parietal before being sent to motor regions of the brain • The prefrontal cortex and premotor cortex are critical for creating a movement plan Test Yourself! An interactive H5P element has been excluded from this version of the text. You can view it online here: https://openbooks.lib.msu.edu/neuroscience/?p=661#h5p-25 Video Version of Lesson A YouTube element has been excluded from this version of the text. You can view it online here: https://openbooks.lib.msu.edu/neuroscience/?p=661 References Roland PE, Larsen B, Lassen NA, Skinhøj E. Supplementary motor area and other cortical areas in organization of voluntary movements in man. J Neurophysiol. 1980 Jan;43(1):118-36. doi: 10.1152/jn.1980.43.1.118. PMID: 7351547. Weinrich M, Wise SP. The premotor cortex of the monkey. J Neurosci. 1982 Sep;2(9):1329-45. doi: 10.1523/JNEUROSCI.02-09-01329.1982. PMID: 7119878; PMCID: PMC6564318."
# out = ollama_gen_soc_question(content2)
# print(out)