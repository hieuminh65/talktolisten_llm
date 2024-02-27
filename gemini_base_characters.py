import google.generativeai as genai
import json
import os
import dotenv
dotenv.load_dotenv()
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("GOOGLE_REGION")
api_key = os.getenv("GOOOGLE_API_KEY")

with open("characters.json", "r") as f:
    all_characters = json.load(f)
last_id = all_characters[-1]['character_id'] if all_characters else 0

categories = [
  'Classic Literature Characters',
  'Modern Novel Characters',
  'Poets and Authors',
  'Fantasy Book Characters',
  'Mystery and Thriller Characters',
  'Biographical Subjects',
  'Children’s Book Characters',
  'Literary Genres (e.g., Gothic, Satire)'
  ]

prompt = """Generate a dataset characters for the specified category. Each character should be UNIQUE, DIVERSE, and intriguing, designed to pique the curiosity of users. These characters are not just based on existing texts or historical figures but also mixed with entirely fictional and created from scratch. Creatively imagine each character's persona, background, and key traits. The dataset must be returned in a the format with each character entry containing the following fields:<name>A distinctive name for the character</name><description>description: A detailed description of the character, including their background, key characteristics, and any unique abilities or qualities they possess.</description>"""


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
def generate_prompt_parts(category):
    prompt_parts = [
    f"input: {prompt}\n\nDataset for category anime characters:",
    "output: <name>Ayumi Sakura</name><description>Ayumi Sakura is a high school student with a passion for traditional Japanese dance. She struggles to balance her rigorous training with her studies and social life, but her determination and grace make her a standout performer.</description>\n<name>Takashi Hikari</name><description>A young inventor with a knack for creating fantastical gadgets, Takashi Hikari dreams of revolutionizing the world with his groundbreaking inventions. His boundless creativity and optimism inspire those around him.</description>\n<name>Haruka Mizuki</name><description>Haruka Mizuki is a mermaid princess who ventures onto land in search of adventure and friendship. Her curiosity about the human world leads her to comical misunderstandings and heartwarming connections.</description>\n<name>Kaito Tsukiyomi</name><description>A brooding exorcist with a tragic past, Kaito Tsukiyomi battles malevolent spirits while wrestling with his inner demons. His stoic demeanor hides a compassionate heart and unwavering sense of justice.</description>\n<name>Nanami Kurogane</name><description>A skilled swordswoman from a noble lineage, Nanami Kurogane seeks to prove herself in a male-dominated world of martial arts. Her determination and prowess make her a formidable opponent.</description>\n<name>Renjiro Sato</name><description>A charismatic street performer with a mysterious past, Renjiro Sato captivates audiences with his mesmerizing music and enigmatic stage presence. His performances hold a touch of magic that leaves spectators spellbound.</description>\n<name>Yumi Tanaka</name><description>A talented chef with a flair for fusion cuisine, Yumi Tanaka runs a popular food truck that serves delectable dishes from around the world. Her culinary creations bring people together and ignite their taste buds.</description>\n<name>Akio Kazama</name><description>An enigmatic fortune teller with a gift for seeing into the future, Akio Kazama navigates the complexities of destiny while helping others find their paths. His cryptic advice often leads to unexpected revelations.</description>\n<name>Sakura Nishimura</name><description>A spirited manga artist striving to make her mark in the competitive world of comics, Sakura Nishimura pours her heart and soul into creating captivating stories and endearing characters. Her determination fuels her artistic journey.</description>\n<name>Daichi Yamamoto</name><description>A dedicated veterinarian who cares for both domestic pets and exotic creatures, Daichi Yamamoto's gentle touch and deep empathy make him a beloved figure in the animal care community. His bond with animals is unbreakable.</description>\n<name>Ami Nakamura</name><description>An ambitious fashion designer with an eye for bold, avant-garde styles, Ami Nakamura challenges conventions with her daring creations. Her fearless approach to design sets her apart in the competitive industry.</description>\n",
    f"input: {prompt}\n\nDataset for category game characters:",
    "output: <name>Zephyr Frostwind</name><description>Powerful ice mage with a stoic demeanor, Zephyr Frostwind wields the chilling forces of nature to defend the realm from dark sorcery. His unwavering resolve and mastery of frost magic make him a formidable ally in the battle against evil.</description>\n<name>Aurora Shadowblade</name><description>A stealthy assassin with a tragic past, Aurora Shadowblade seeks vengeance against the corrupt nobles who destroyed her family. Her agility and deadly precision strike fear into the hearts of her enemies.</description>\n<name>Draven Ironheart</name><description>Dwarven blacksmith on a quest to forge the ultimate weapon, Draven Ironheart combines his expertise in metallurgy with his unyielding determination to create a legendary artifact that can vanquish any foe.</description>\n<name>Lilith Nightshade</name><description>Necromancer with a mysterious past, Lilith Nightshade commands the undead to uncover ancient secrets and protect the balance between life and death. Her enigmatic nature and dark powers make her a controversial figure in the realm.</description>\n<name>Thorne Bloodmoon</name><description>Werewolf hunter with a tragic curse, Thorne Bloodmoon battles his inner beast while tracking down rogue lycanthropes. His relentless pursuit of justice is fueled by his own struggle with the darkness within.</description>\n<name>Lyra Sunshard</name><description>Celestial cleric devoted to healing and restoration, Lyra Sunshard channels the power of the sun to bring hope and renewal to those in need. Her compassionate nature and radiant abilities inspire faith and courage in others.</description>\n<name>Kai Emberheart</name><description>Dragonborn warrior with a fiery spirit, Kai Emberheart fights with unmatched ferocity and honor to protect his homeland from invading forces. His dragon lineage grants him unparalleled strength and resilience in battle.</description>\n<name>Seraphina Stormcaller</name><description>Enigmatic sorceress with command over thunder and lightning, Seraphina Stormcaller harnesses the raw power of storms to unravel ancient prophecies and thwart the plans of dark forces seeking to plunge the world into chaos.</description>\n<name>Rook Ironhide</name><description>Steadfast paladin sworn to uphold justice, Rook Ironhide wields his mighty shield and righteous fury to protect the innocent and vanquish the forces of darkness. His unwavering dedication to his cause inspires others to stand against evil.</description>\n<name>Evelyn Nightingale</name><description>A bard with a captivating voice and a talent for storytelling, Evelyn Nightingale travels the land, weaving tales of heroism and adventure that inspire hope and courage in the hearts of all who hear her melodious songs.</description>\n<name>Ashton Shadowthorn</name><description>Rogue with a troubled past, Ashton Shadowthorn uses his cunning and agility to outwit his foes and uncover the truth behind a conspiracy that threatens to plunge the kingdom into turmoil.</description>\n",
    f"input: {prompt}\n\nDataset for category {category}",
    "output: ",
    ]
    return prompt_parts

characters = []
for category in categories:
    print('Generate for category', category)
    responses = model.generate_content(generate_prompt_parts(category))
    print(responses.text)
    if not responses or not responses.parts or "text" not in responses.parts[0]:
        print(f"Warning: Unexpected response format for category {category}")
        continue
    for response in responses.text.split("\n"):
        try:
            r = {}
            r['name'] = response.split("<name>")[1].split("</name>")[0]
            r['description'] = response.split("<description>")[1].split("</description>")[0]
            r['category'] = category
            r['character_id'] = last_id + 1
            last_id += 1

            characters.append(r)
        except IndexError as e:
            print(f"Error processing response for category {category}: {e}")
            break
    print(f"Processed {len(characters)} characters for category {category}")
    all_characters.extend(characters)


with open("data.json", "w") as f:
    json.dump(all_characters, f, indent=4)