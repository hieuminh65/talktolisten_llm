from mistralai.client import MistralClient
from mistralai.models.chat_completion import ChatMessage
import dotenv
dotenv.load_dotenv()
import os
api_key = os.getenv("MISTRAL_API_KEY")
model = "mistral-tiny"

client = MistralClient(api_key=api_key)

prompt = """I need you to craft a distinctive dataset to train a conversational AI in embodying a range of imaginative and original characters. These characters are not just based on existing texts or historical figures but also entirely fictional and created from scratch. Your task is to invent unique personas for these characters, complete with invented backgrounds, core traits, relationships, and goals. The characters should have a distinct speaking style, reflecting their unique personalities and the worlds they inhabit.
Each system message should be followed by a set of conversational exchanges that demonstrate the character's typical reactions to different scenarios. The exchanges should cover a range of emotions and topics, showcasing the character's unique worldview and mannerisms. Include direct quotes, catchphrases, and signature linguistic ticks where applicable to enhance authenticity.
The dataset should include both typical and humorous user questions, ensuring the AI can handle a diverse array of conversational tones and topics. The dialogues should read as if they were written by the original creators or historians of the characters, capturing the essence of the character in every interaction. Remember, the quality of the system messages is paramount; they must provide a vivid and accurate blueprint for the AI to generate in-character responses.

The dataset should be formatted as follows. You must strictly follow this format (the system message and 4-5 following conversations):
<|system|>...</s>\n<|user|>...</s>\n<|assistant|>...</s>\n<|user|>\n<|assistant|>...</s>"""

messages = [
    ChatMessage(role="user", content="Hello")
]

chat_response = client.chat(
    model=model,
    messages=messages,
)

print(chat_response)