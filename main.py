from fastapi import FastAPI, HTTPException
from transformers import pipeline
from pydantic import BaseModel

class Item(BaseModel):
    text: str

app = FastAPI()
classifier = pipeline("sentiment-analysis")

@app.get("/")
def root():
    """Корневой маршрут возвращает сообщение "Hello World"."""
    return {"message": "Hello World"}

@app.post("/predict/")
def predict(item: Item):
    """Принимает JSON с текстом и возвращает предсказание настроения."""
    try:
        result = classifier(item.text)[0]
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail="Ошибка при обработке запроса")
