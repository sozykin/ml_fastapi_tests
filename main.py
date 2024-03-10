from fastapi import FastAPI
from transformers import pipeline
from pydantic import BaseModel


class Item(BaseModel):
    text: str


app = FastAPI()
model_id = "cardiffnlp/twitter-roberta-base-sentiment-latest"

classifier = pipeline("sentiment-analysis", model=model_id)


@app.get("/")
def root():
    return {"message": "Hello World"}


@app.post("/predict/")
def predict(item: Item):
    return classifier(item.text)[0]
