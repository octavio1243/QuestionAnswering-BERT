import torch
from fastapi import FastAPI
from pydantic import BaseModel
import torch
from transformers import  BertTokenizer, BertForQuestionAnswering
from unidecode import unidecode
import os
import uvicorn

class PredictRequest(BaseModel):
    text: str
    question: str

def load_model_tokenizer():
    ruta_script = os.path.abspath(__file__)
    directorio_actual = os.path.dirname(os.path.realpath(__file__))
    path  = os.path.join(directorio_actual, "dataset")
    
    model = BertForQuestionAnswering.from_pretrained(path)
    tokenizer = BertTokenizer.from_pretrained(path, do_lower_case=False)
    model.eval()

    return model, tokenizer

def preprocess_text(text):
    # Quitar acentos
    text_without_accents = unidecode(text)
    # Convertir a minúsculas
    text_lower = text_without_accents.lower()

    return text_lower

def get_answer(question, text):
    global tokenizer
    global e

    question = preprocess_text(question)
    text = preprocess_text(text)

    # Tokenizar la question y el text
    inputs = tokenizer(question, text, return_tensors="pt")

    # Obtener la salida del modelo
    output = model(**inputs)

    # Obtener los índices de inicio y fin de la respuesta
    answer_start = torch.argmax(output.start_logits)
    answer_end = torch.argmax(output.end_logits) + 1

    # Obtener el texto de la respuesta
    answer = tokenizer.decode(inputs["input_ids"][0][answer_start:answer_end])

    return answer

VERSION = "0.1.0"
print(f"Loading model version {VERSION}")
model, tokenizer  = load_model_tokenizer()
print("Model ready")

"""
pregunta = "¿Cuál es la capital de España?"
contexto = "España es un país ubicado en el suroeste de Europa. Limita al norte con Francia, al este con el mar Mediterráneo, al sur con el mar de Alborán y al oeste con Portugal. Su capital es Madrid, que es también la ciudad más grande del país y un importante centro cultural y económico."
answ =get_answer(pregunta, contexto)
print(answ)
"""

app = FastAPI()

@app.post("/model/predict")
def predict(predict_request: PredictRequest):
    answer = get_answer(predict_request.question, predict_request.text)
    return {"answer":answer}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)