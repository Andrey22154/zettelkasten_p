import torch
from transformers import BertModel, BertTokenizer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from typing import List


MODEL_PATH = "path"  # Укажите путь к папке с вашей обученной моделью
MAX_LEN = 320  # Укажите максимальную длину, которую вы использовали при обучении
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained(MODEL_PATH)
model.to(DEVICE)

app = FastAPI()

class NoteRequest(BaseModel):
    note: str

def preprocess_text(text, max_len):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt',
    )
    return inputs.to(DEVICE)

@app.post("/find_similar", response_model=List[str])
def find_similar_notes(note_request: NoteRequest):
    input_note = note_request.note
    input_note_preprocessed = preprocess_text(input_note, MAX_LEN)

    with torch.no_grad():
        outputs = model(**input_note_preprocessed)
        input_embedding = outputs.last_hidden_state[:, 0, :]

    data = pd.read_csv('path')  
    dataset = data.title.values + data.tags.values + data['main text'].values
    similarities = []

    for note in dataset:
        note_preprocessed = preprocess_text(note, MAX_LEN)
        with torch.no_grad():
            note_outputs = model(**note_preprocessed)
            note_embedding = note_outputs.last_hidden_state[:, 0, :]
        cos_sim = torch.nn.functional.cosine_similarity(input_embedding, note_embedding)
        similarities.append(cos_sim.item())

    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:3]
    top_notes = [dataset[i] for i in top_indices if similarities[i] > 0.9]

    return top_notes

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
