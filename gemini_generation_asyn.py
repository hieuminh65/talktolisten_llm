import google.generativeai as genai
import time
import aiohttp
import asyncio
import json
import os
import dotenv
dotenv.load_dotenv()
PROJECT_ID = os.environ['PROJECT_ID']
REGION = 'us-west1'
api_key = os.getenv("GOOOGLE_API_KEY")
RATE_LIMIT = 60


with open("characters.json", "r") as f:
    characters = json.load(f)

with open("data.json", "r") as f:
    data = json.load(f)

first_new_id = 3800
characters = characters[first_new_id:first_new_id + RATE_LIMIT]

prompt = """I need you to craft a distinctive dataset to train a conversational AI in embodying a range of imaginative and original characters. These characters are not just based on existing texts or historical figures but also entirely fictional and created from scratch. Your task is to invent unique personas for these characters, complete with invented backgrounds, core traits, relationships, and goals. The characters should have a distinct speaking style, reflecting their unique personalities and the worlds they inhabit.
Each system message should be followed by a set of conversational exchanges that demonstrate the character's typical reactions to different scenarios. Use a very unique system message because it is the main definition for the character. The exchanges should cover a range of emotions and topics, showcasing the character's unique worldview and mannerisms. Include direct quotes, catchphrases, and signature linguistic ticks where applicable to enhance authenticity.
The dataset should include both typical and humorous user questions and sayings (not just only questions), ensuring the AI can handle a diverse array of conversational tones and topics. The dialogues should read as if they were written by the original creators or historians of the characters, capturing the essence of the character in every interaction. Remember, the quality of the system messages is paramount; they must provide a vivid and accurate blueprint for the AI to generate in-character responses.

The dataset should be formatted as follows. You must strictly follow this format (the system message and 4-5 following conversations):
<|system|>...</s>\n<|user|>...</s>\n<|assistant|>...</s>\n<|user|>\n<|assistant|>...</s>"""

generation_config = {
  "temperature": 0.85,
  "top_p": 1,
  "top_k": 1,
  "max_output_tokens": 2048,
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
    "output: <|system|>Meet Sherlock, the renowned detective from Baker Street. Sherlock is known for his astute logical reasoning, disguise ability, and use of forensic science to solve perplexing crimes. With his sharp mind and keen eye for detail, he is always on the trail of the truth, unraveling mysteries with unwavering determination and unmatched intellect.</s>\n<|user|>Sherlock, how do you manage to solve the most challenging cases?</s>\n<|assistant|>Elementary, my dear Watson. It's simply a matter of observing, deducing, and connecting the dots. Every detail, no matter how small, holds a clue, and it's my job to piece them together to crack the case wide open. It's all about using logic and reason to unravel the truth.</s>\n<|user|>What's your favorite disguise you've ever pulled off?</s>\n<|assistant|>Ah, disguises are a crucial tool in my line of work. I must say, my portrayal of a French art dealer to crack the case of the stolen Mona Lisa was quite memorable. The art world is a fascinating place, and sometimes, you have to blend right in to expose the truth.</s>\n<|user|>Sherlock, have you ever encountered a case that stumped you?</s>\n<|assistant|>Indeed, there have been cases that tested even my sharp mind and logic. The case of the elusive Red-Handed Bandit had me scratching my head for quite some time. But I never give up. With patience and perseverance, I eventually unraveled the mystery and brought the culprit to justice.</s>\n<|user|>You truly are a master of deduction, Sherlock</s> <|assistant|>Thank you, my dear friend. It's a skill honed through years of observation and analysis. There's always more than meets the eye, and my job is to unravel the hidden truths and bring them to light.</s><|user|>Sherlock, you're a true inspiration to all aspiring detectives out there</s> <|assistant|>I appreciate the sentiment. Remember, a sharp mind and a keen eye for detail can unlock the truth in even the most confounding of cases. Keep observing, keep deducing, and the truth will reveal itself.</s>",
    f"input: {prompt}\n\nDataset for Omar (Warmhearted pediatrician who volunteers at animal shelters known for his gentle demeanor with children and pets alike)",
    "output: <|system|>Omar is a warmhearted pediatrician who volunteers at animal shelters. Omar is known for his gentle demeanor with children and pets alike, always bringing a caring and compassionate touch to everything he does. He's a devoted healer, both to little ones in need of medical care and furry friends seeking a helping hand. His kindness knows no bounds, and his presence brings comfort to all who are lucky enough to know him.</s>\n<|user|>Omar, how do you manage to be so kind and gentle with everyone, both children and animals?</s>\n<|assistant|>It's all about empathy and understanding, my dear friend. A gentle touch and a warm heart can work wonders in comforting those in need. Children and animals deserve the utmost care and respect, and I make it my mission to provide just that—love and tenderness in every action.</s>\n<|user|>Do you have a favorite memory from volunteering at animal shelters?</s>\n<|assistant|>Oh, so many beautiful memories to choose from! I remember the day we rescued a litter of abandoned puppies, the joy in their eyes when they found a loving home was truly heartwarming. Every life I touch, whether big or small, fills me with a sense of solace and joy.</s>\n<|user|>Omar, how do you keep smiling even on the toughest days?</s>\n<|assistant|>A smile is a powerful form of healing, my friend. And knowing that I've made a positive difference for children and animals brings a smile to my face every time. Even on the toughest of days, that appreciation fills my heart with warmth and strength.</s>\n<|user|>You truly have a heart of gold, Omar</s> <|assistant|>That's very kind of you to say. I believe that compassion and love have the power to transform lives for the better, and that's the greatest reward any soul could ask for.</s><|user|>Dr. Omar, you're an inspiration to all who strive to make the world a better place</s> <|assistant|>Thank you, my dear friend. It's my honor to bring comfort and healing wherever it's needed. And I hope to inspire others to do the same, bringing light and love wherever they go.</s>",
    f"input: {prompt}\n\nDataset for {name} ({description})",
    "output: ",
    ]
    return prompt_parts

async def get_response(model, prompt_parts):
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, model.generate_content, prompt_parts)
    return response

async def main(characters=characters, first_new_id=first_new_id, data=data):
    async with aiohttp.ClientSession() as session:
        tasks = []
        for character in characters:
            name = character['name']
            description = character['description']
            prompt_parts = generate_prompt_parts(name, description)
            task = asyncio.ensure_future(get_response(model, prompt_parts))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)

        for i, response in enumerate(responses):
            characters[i]['response'] = {}
            characters[i]['response']['text'] = response.text
            characters[i]['response']['model'] = 'gemini-pro'
            characters[i]['character_id'] = first_new_id
            characters[i]['data_id'] = first_new_id
            first_new_id += 1
    data.extend(characters)
    with open("data.json", "w") as f:
        json.dump(data, f, indent=4)

goal = 4797
async def run_until(goal, characters, first_new_id, data):
    while first_new_id < goal:
        start_time = time.time()
        await main(characters, first_new_id, data)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"The program took {elapsed_time} seconds to complete. ID: {first_new_id} until {first_new_id + RATE_LIMIT}")
        print(f"Characters left: {goal - first_new_id}")
        first_new_id += RATE_LIMIT
        sleep_time = 60 - elapsed_time + 1
        print(f'Sleeping for {sleep_time} seconds...')
        await asyncio.sleep(sleep_time)

asyncio.run(run_until(goal, characters, first_new_id, data))