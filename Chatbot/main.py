from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage, SystemMessage
from PIL import Image
import io
import os
import requests

app = FastAPI()

load_dotenv()

model = ChatMistralAI(
    model='pixtral-12b-2409',
    temperature=0.5,
    max_retries=2,
)

@app.post('/predict_disease')
async def predict_disease(
    symptoms: str = Form(...),
    image: UploadFile = File(None)
):
    image_content = ''
    if image:
        try:
            contents = await image.read()
            img = Image.open(io.BytesIO(contents))
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='PNG')
            image_content = img_byte_arr.getvalue()
        except Exception as e:
            raise HTTPException(status_code=400, detail=f'Error processing image: {str(e)}')

    messages = [
        SystemMessage(content='Predict the disease based on these symptoms and the provided image (if any). Give the medicines one can take to prevent this:'),
        HumanMessage(content=symptoms, additional_kwargs={'image': image_content} if image_content else {}),
    ]

    try:
        result = model.invoke(messages)
        return JSONResponse(content={'prediction': result.content})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Error during prediction: {str(e)}')

@app.get('/')
async def root():
    return {'message': 'Welcome to the Disease Predictor API'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
