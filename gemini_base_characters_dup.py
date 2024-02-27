import google.generativeai as genai
import aiohttp
import asyncio
import json
import os
import dotenv
from random import randint
dotenv.load_dotenv()
PROJECT_ID = os.environ['PROJECT_ID']
REGION = 'us-west1'
api_key = os.getenv("GOOOGLE_API_KEY")

with open("characters2.json", "r") as f:
    all_characters = json.load(f)
last_id = all_characters[-1]['character_id'] if all_characters else 0


prompt1 = """Generate a dataset characters for the specified category. Each character should be UNIQUE, DIVERSE, and intriguing, designed to pique the curiosity of users. These characters are not just based on existing texts or historical figures but also mixed with entirely fictional and created from scratch. Creatively imagine each character's persona, background, and key traits. The dataset must be returned in a the format with each character entry containing the following fields:<name>A distinctive name for the character</name><description>description: A detailed description of the character, including their background, key characteristics, and any unique abilities or qualities they possess.</description>"""
prompt2 = """Create a character dataset for the given category. Every character should be engaging, varied, and unique in order to stimulate users' curiosity. These characters are wholly fictitious and have been created from scratch, in addition to being based on previously published works or historical persons. Imaginatively create each character's individuality, history, and essential characteristics. The dataset must be returned in a the format with each character entry containing the following fields:<name>A distinctive name for the character</name><description>description: A detailed description of the character, including their background, key characteristics, and any unique abilities or qualities they possess.</description>"""
prompt3 = """Produce a character dataset for the designated category. Aiming to stimulate consumers' curiosity, every character ought to be intriguing, diverse, and unique. Along with being completely fictitious and made from scratch, these characters are not only based on books that already exist or historical persons. Imagine in a unique way the persona, history, and essential characteristics of each character. The dataset must be returned in a the format with each character entry containing the following fields:<name>A distinctive name for the character</name><description>description: A detailed description of the character, including their background, key characteristics, and any unique abilities or qualities they possess.</description>"""
prompt4 = """For the given category, create a dataset of characters. To arouse consumers' curiosity, every character ought to be UNIQUE, DIVERSE, and captivating. These characters are completely fictional and made from scratch, in addition to being based on works of literature that already exist or historical figures. Imagine each character's persona, history, and essential characteristics in a creative way. The dataset must be returned in a the format with each character entry containing the following fields:<name>A distinctive name for the character</name><description>description: A detailed description of the character, including their background, key characteristics, and any unique abilities or qualities they possess.</description>"""
prompt5 = """Construct an assortment of captivating figures tailored to the provided category, ensuring each character possesses a distinct identity, a tapestry of diversity, and an irresistible allure to engage users. Blend elements of reality, fantasy, and originality in crafting these characters, infusing each with a compelling narrative, unique attributes, and an enigmatic charm to captivate the imagination. The dataset must be returned in a the format with each character entry containing the following fields:<name>A distinctive name for the character</name><description>description: A detailed description of the character, including their background, key characteristics, and any unique abilities or qualities they possess.</description>"""
prompt6 = """Develop a collection of intriguing personas fitting the specified category, guaranteeing each character stands out as a singular entity, diverse in nature, and brimming with captivating qualities to enthrall users. These characters should meld real-world influences, imaginative origins, and innovative concepts, each characterized by a compelling backstory, distinctive traits, and an aura of mystique to captivate the audience. The dataset must be returned in a the format with each character entry containing the following fields:<name>A distinctive name for the character</name><description>description: A detailed description of the character, including their background, key characteristics, and any unique abilities or qualities they possess.</description>"""
prompt7 = """Curate a selection of captivating individuals that align with the specified category, guaranteeing each character exudes uniqueness, diversity, and an irresistible allure to captivate users' interest. Integrate a blend of factual inspirations, fantastical elements, and inventive concepts to shape these personas, imbuing each with a captivating backstory, defining characteristics, and an aura of intrigue to captivate and enchant the audience. The dataset must be returned in a the format with each character entry containing the following fields:<name>A distinctive name for the character</name><description>description: A detailed description of the character, including their background, key characteristics, and any unique abilities or qualities they possess.</description>"""
list_of_prompts = [prompt1, prompt2, prompt3 ,prompt4,prompt5,prompt6,prompt7]
genai.configure(api_key=api_key)


generation_config = {
  "temperature": 1,
  # "top_p": 1,
  "top_k": 50,
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

model = genai.GenerativeModel(model_name="gemini-pro",
                              generation_config=generation_config,
                              safety_settings=safety_settings)
def generate_prompt_parts(prompt, dataset_1, dataset_2):
    prompt_parts = [
    f"input: {prompt}\n\n",
    f"output: {dataset_1}",
    f"input: {prompt}\n\n",
    f"output: {dataset_2}",
    f"input: {prompt}\n\n",
    "output: ",
    ]
    return prompt_parts

characters = []

def choose_random_prompt(list_of_prompts):
    index = randint(0, len(list_of_prompts)-1)
    the_chosen_prompt = list_of_prompts[index]
    return the_chosen_prompt  

def choose_random_data(all_characters):
    list_of_choosen_characters = []
    final_list_of_10_example = []
    for i in range(0,11):
        i = randint(0,len(all_characters)-1)
        list_of_choosen_characters.append(all_characters[i])
    for character in list_of_choosen_characters:
        concat_data = ""
        concat_data += '<name>'
        concat_data += character['name']
        concat_data += '</name>'
        concat_data += '<description>'
        concat_data += character['description']
        concat_data += '</description>'
        concat_data += '\n'
        final_list_of_10_example.append(concat_data)
    return "".join(final_list_of_10_example)

for i in range(2):
    prompt_for_gen = choose_random_prompt(list_of_prompts)
    dataset1 = choose_random_data(all_characters)
    dataset2 = choose_random_data(all_characters)
    responses = model.generate_content(generate_prompt_parts(prompt_for_gen, dataset1, dataset2))
    if not responses or not responses.parts or "text" not in responses.parts[0]:
        print(f"Warning: Unexpected response format.")
        continue
    for response in responses.text.split("\n"):
        try:
            r = {}
            r['name'] = response.split("<name>")[1].split("</name>")[0]
            r['description'] = response.split("<description>")[1].split("</description>")[0]
            r['character_id'] = last_id + 1
            last_id += 1

            characters.append(r)
        except IndexError as e:
            print(f"Error processing response: {e}")
            break
    print(f"Processed {len(characters)} characters")
    all_characters.extend(characters)

with open("characters3.json", "w") as f:
    json.dump(all_characters, f, indent=4)