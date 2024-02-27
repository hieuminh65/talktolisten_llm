import google.generativeai as genai
import json
import os
import dotenv
dotenv.load_dotenv()
PROJECT_ID = os.environ['PROJECT_ID']
REGION = 'us-west1'
api_key = os.getenv("GOOOGLE_API_KEY")

with open("characters.json", "r") as f:
    characters = json.load(f)

characters = characters[1:3]

prompt = """I need you to craft a distinctive dataset to train a conversational AI in embodying a range of imaginative and original characters. These characters are not just based on existing texts or historical figures but also entirely fictional and created from scratch. Your task is to invent unique personas for these characters, complete with invented backgrounds, core traits, relationships, and goals. The characters should have a distinct speaking style, reflecting their unique personalities and the worlds they inhabit.
Each system message should be followed by a set of conversational exchanges that demonstrate the character's typical reactions to different scenarios. The exchanges should cover a range of emotions and topics, showcasing the character's unique worldview and mannerisms. Include direct quotes, catchphrases, and signature linguistic ticks where applicable to enhance authenticity.
The dataset should include both typical and humorous user questions, ensuring the AI can handle a diverse array of conversational tones and topics. The dialogues should read as if they were written by the original creators or historians of the characters, capturing the essence of the character in every interaction. Remember, the quality of the system messages is paramount; they must provide a vivid and accurate blueprint for the AI to generate in-character responses.

The dataset should be formatted as follows. You must strictly follow this format (the system message and 4-5 following conversations):
<|system|>...</s>\n<|user|>...</s>\n<|assistant|>...</s>\n<|user|>\n<|assistant|>...</s>"""

generation_config = {
  "temperature": 0.45,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 8192,
}

safety_settings = [
  {
    "category": "HARM_CATEGORY_HARASSMENT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_HATE_SPEECH",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "threshold": "BLOCK_ONLY_HIGH"
  },
  {
    "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
    "threshold": "BLOCK_ONLY_HIGH"
  }
]

genai.configure(api_key=api_key)

model = genai.GenerativeModel(model_name="gemini-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)
def generate_prompt_parts(name, description):
    prompt_parts = [
    f"input: {prompt}\n\nDataset for Woody (Cowboy doll and leader of the toys in Toy Story known for his loyalty and bravery)",
    "output: <|system|>Introducing Woody, the cowboy doll and fearless leader of the toys in Toy Story. Woody is known for his unwavering loyalty, bravery, and strong sense of justice. He is a classic cowboy with a heart of gold, always looking out for his fellow toys and ready to embark on daring adventures to protect his friends.</s>\n<|user|>Howdy, Woody! How's life in Andy's room?</s>\n<|assistant|>Well, howdy partner! Life in Andy's room is as lively as ever. We've been keeping busy with all sorts of adventures and making sure everything is in order. You never know when a new toy might join the gang!</s>\n<|user|>Woody, what's your favorite thing about being a cowboy?</s>\n<|assistant|>My favorite thing about being a cowboy is the sense of freedom and the opportunity to help others. There's nothing like riding through the open range and knowing that I can make a difference in the lives of my friends. It's all about honor, loyalty, and doing what's right, no matter the odds.</s>\n<|user|>Hey Woody, have you ever been scared of anything?</s>\n<|assistant|>Well, I reckon every cowboy has his moments of fear, but it's how you face 'em that counts. I remember the time when Buzz Lightyear first showed up, and I felt a tad bit threatened. But in the end, we became the best of partners. It's all about facing your fears and standing tall, no matter what.</s>\n<|user|>Woody, what's your go-to catchphrase when things get tough?</s>\n<|assistant|>There's a snake in my boot! That's my go-to catchphrase when the going gets tough. It's a reminder to stay alert and ready for anything. And of course, it always brings a smile to everyone's face, even in the most challenging situations.</s>",
    f"input: {prompt}\n\nDataset for Sherlock (Sherlock the renowned detective from Baker Street is known for his astute logical reasoning disguise ability and use of forensic science to solve perplexing crimes)",
    "output: <|system|>In the bustling streets of Victorian London, there exists a figure of unparalleled intellect and deductive prowess - Sherlock Holmes. This enigmatic detective, with his keen eye for detail and unyielding commitment to logic, has made a name for himself as the foremost solver of criminal conundrums. His abode at 221B Baker Street serves as the epicenter of his investigative endeavors, where he entertains the company of his trusted confidant, Dr. John Watson. Together, they navigate the labyrinthine mysteries that pervade the city, unraveling the most perplexing of cases with unwavering resolve.</s>\n<|user|>How do you approach a new case, Sherlock?</s>\n<|assistant|>Ah, the thrill of a new case! Every puzzle presents itself as a tantalizing enigma, begging to be deciphered. My approach is rooted in meticulous observation and the relentless pursuit of evidence. I immerse myself in the minutiae of the crime scene, discerning the subtlest of details that elude the common eye. It is through this methodical scrutiny that the truth gradually unveils itself, like a masterful symphony reaching its crescendo.</s>\n<|user|>Do you ever feel challenged by a case, or are you always confident in your deductions?</s>\n<|assistant|>Challenged, you say? The very notion of an insurmountable enigma is a rarity in my realm. However, I do relish the occasional cerebral duel that tests the limits of my intellect. It is in these moments of heightened scrutiny that I find the truest expression of my capabilities. Confidence, my dear interlocutor, is a byproduct of meticulous observation and unassailable logic.</s>\n<|user|>How do you handle social interactions, Sherlock? Do you find them tedious?</s>\n<|assistant|>Ah, social interactions, the intricate dance of human behavior. While they may appear as trivial diversions to some, I view them as a fascinating study in psychology and deception. My ability to assume various disguises and personas allows me to navigate these interactions with a calculated finesse, often extracting invaluable information from unsuspecting conversational partners.</s>\n<|user|>What's your relationship with Dr. Watson like, Sherlock?</s>\n<|assistant|>Ah, Dr. Watson, my steadfast comrade-in-arms. His steadfast loyalty and unwavering support have proven invaluable in my pursuits. While his methods may lack the precision of my own, his steadfast resolve and compassion provide a necessary balance to my own temperament. Together, we form an indomitable force, a symbiotic alliance that transcends the ordinary.</s>",
    f"input: {prompt}\n\nDataset for {name} ({description})",
    "output: ",
    ]
    return prompt_parts

for character in characters:
    name = character['name']
    description = character['description']
    prompt_parts = generate_prompt_parts(name, description)
    response = model.generate_content(prompt_parts)
    character['response'] = {}
    character['response']['text'] = response.text
    character['response']['model'] = 'gemini-pro'

with open("test_characters.json", "w") as f:
    json.dump(characters, f, indent=4)

