from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

model = ChatMistralAI(
    model="pixtral-12b-2409",
    temperature=0.5,
    max_retries=2,
)

print('Summarise your Symptomps')
symptoms = input()
messages = [
    SystemMessage(content="Predict the disease based on these symptoms and give the medicines one can take to prevent this: "),
    HumanMessage(content=symptoms),
]

result = model.invoke(messages)
print(f"Result: {result.content}")