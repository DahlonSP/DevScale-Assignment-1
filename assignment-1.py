import os
import fitz

from dotenv import load_dotenv
from openai import BaseModel, OpenAI


load_dotenv()

SUMOPOD_API_KEY = os.getenv("SUMOPOD_API_KEY")
SUMOPOD_BASE_URL = os.getenv("SUMOPOD_BASE_URL")

doc = fitz.open("FAA 2026-09-01.pdf")
for i, page in enumerate(doc):
    SYSTEM_PROMPT = page.get_text()

client = OpenAI(base_url=SUMOPOD_BASE_URL, api_key=SUMOPOD_API_KEY)

class AirworthinessDirective (BaseModel):
    ad_no : str
    effective_date : str
    affected_ads: str
    unsafe_condition: str

messages=[
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": "Hello!",
        }
    ]

while True:
    user_input = input("User: ")
    user_mesage = {
        "role": "user",
        "content": user_input,
    }

    messages.append(user_mesage)

    completion = client.chat.completions.parse(
    model="nvidia/nemotron-3-nano-30b",
    messages= messages,
    response_format=AirworthinessDirective
    )


    final_output = completion.choices[0].message.content or ""
    print(final_output)


    messages.append({
        "role": "assistant",
        "content": final_output,
    })
