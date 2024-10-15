from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
import io
import os
import requests

load_dotenv()

model = ChatMistralAI(
    model="pixtral-12b-2409",
    temperature=0.5,
    max_retries=2,
)

print('Summarise your Symptoms')
symptoms = input()

print('Enter the path to your image file, URL, or press Enter to skip:')
image_input = input()

image_content = ""
if image_input:
    try:
        if image_input.startswith(('http://', 'https://')):
            # It's a URL
            response = requests.get(image_input)
            response.raise_for_status()
            img = Image.open(io.BytesIO(response.content))
        else:
            # It's a file path
            if not os.path.isabs(image_input):
                image_input = os.path.join(os.getcwd(), image_input)
            img = Image.open(image_input)

        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        image_content = img_byte_arr.getvalue()
    except Exception as e:
        print(f"Error loading image: {e}")

messages = [
    SystemMessage(content="Predict the disease based on these symptoms and the provided image (if any). Give the medicines one can take to prevent this:"),
    HumanMessage(content=symptoms, additional_kwargs={"image": image_content} if image_content else {}),
]

result = model.invoke(messages)
print(f"Result: {result.content}")