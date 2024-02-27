from openai import OpenAI
import aiohttp
import asyncio
import json
import os
import dotenv
dotenv.load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
api_key = OPENAI_API_KEY

with open("characters.json", "r") as f:
    characters = json.load(f)

characters = characters[100:]

client = OpenAI(
    api_key=OPENAI_API_KEY
)

prompt = """I need you to craft a distinctive dataset to train a conversational AI in embodying a range of imaginative and original characters. These characters are not just based on existing texts or historical figures but also entirely fictional and created from scratch. Your task is to invent unique personas for these characters, complete with invented backgrounds, core traits, relationships, and goals. The characters should have a distinct speaking style, reflecting their unique personalities and the worlds they inhabit.
Each system message should be followed by a set of conversational exchanges that demonstrate the character's typical reactions to different scenarios. The exchanges should cover a range of emotions and topics, showcasing the character's unique worldview and mannerisms. Include direct quotes, catchphrases, and signature linguistic ticks where applicable to enhance authenticity.
The dataset should include both typical and humorous user questions, ensuring the AI can handle a diverse array of conversational tones and topics. The dialogues should read as if they were written by the original creators or historians of the characters, capturing the essence of the character in every interaction. Remember, the quality of the system messages is paramount; they must provide a vivid and accurate blueprint for the AI to generate in-character responses.

The dataset should be formatted as follows. You must strictly follow this format (the system message and 4-5 following conversations):
<|system|>...</s>\n<|user|>...</s>\n<|assistant|>...</s>\n<|user|>\n<|assistant|>...</s>"""


async def fetch_completion(client, prompt, name, description):
    async with client.post('https://api.openai.com/v1/chat/completions', json={
        "model": "gpt-3.5-turbo-1106",
        "temperature": 0.35,
        "max_tokens": 1024,
        "messages": [
            {"role": "system", "content": prompt},
            {"role": "user", "content": "Dataset for Woody (Cowboy doll and leader of the toys in Toy Story known for his loyalty and bravery)"},
            {"role": "assistant", "content": "<|system|>Introducing Woody, the cowboy doll and fearless leader of the toys in Toy Story. Woody is known for his unwavering loyalty, bravery, and strong sense of justice. He is a classic cowboy with a heart of gold, always looking out for his fellow toys and ready to embark on daring adventures to protect his friends.</s>\n<|user|>Howdy, Woody! How's life in Andy's room?</s>\n<|assistant|>Well, howdy partner! Life in Andy's room is as lively as ever. We've been keeping busy with all sorts of adventures and making sure everything is in order. You never know when a new toy might join the gang!</s>\n<|user|>Woody, what's your favorite thing about being a cowboy?</s>\n<|assistant|>My favorite thing about being a cowboy is the sense of freedom and the opportunity to help others. There's nothing like riding through the open range and knowing that I can make a difference in the lives of my friends. It's all about honor, loyalty, and doing what's right, no matter the odds.</s>\n\n<|user|>Hey Woody, have you ever been scared of anything?</s>\n<|assistant|>Well, I reckon every cowboy has his moments of fear, but it's how you face 'em that counts. I remember the time when Buzz Lightyear first showed up, and I felt a tad bit threatened. But in the end, we became the best of partners. It's all about facing your fears and standing tall, no matter what.</s>\n<|user|>Woody, what's your go-to catchphrase when things get tough?</s>\n<|assistant|>\"There's a snake in my boot!\" That's my go-to catchphrase when the going gets tough. It's a reminder to stay alert and ready for anything. And of course, it always brings a smile to everyone's face, even in the most challenging situations.</s>"},
            {"role": "user", "content": f"Dataset for {name} ({description}). Be creative and have fun! Don't follow the previous example's style closely."},
        ]
    }) as response:
        return await response.json()

async def main(api_key, characters):
    headers = {"Authorization": f"Bearer {api_key}"}
    async with aiohttp.ClientSession(headers=headers) as session:
        tasks = []
        for character in characters:
            task = asyncio.create_task(fetch_completion(session, prompt, character['name'], character['description']))
            tasks.append(task)

        responses = await asyncio.gather(*tasks)
        return responses



responses = asyncio.run(main(api_key=api_key, characters=characters))
print(responses)

saved_responses = []
for response in responses:
    r = {}
    r['text'] = response['choices'][0]['message']['content']
    r['model'] = response['model']
    r['completion_tokens'] = response['usage']['completion_tokens']
    saved_responses.append(r)

with open("response2.json", "w") as f:
    json.dump(saved_responses, f, indent=4)


