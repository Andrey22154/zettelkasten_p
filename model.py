from typing import List
from fastapi import FastAPI
from transformers import DistilBertTokenizer, DistilBertModel
import torch
import uvicorn
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI()

# Загрузка предобученной модели DistilBERT
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Общий датасет для хранения заметок
dataset = []


@app.post("/notes/")
def create_note(note: str):
    # Разделение заметки на заголовок, тэги и текст
    title, tags, text = note.split(".")

    # Разделение тэгов на отдельные теги
    tags = [tag.strip() for tag in tags.split("#") if tag.strip()]

    # Сохранение заметки в общий датасет
    dataset.append((title.strip(), tags, text.strip()))

    return {"message": "Заметка сохранена успешно"}


@app.get("/similar_notes/")
def get_similar_notes(note: str):
    # Разделение заметки на заголовок, тэги и текст
    title, tags, text = note.split(".")

    # Разделение тэгов на отдельные теги
    tags = [tag.strip() for tag in tags.split("#") if tag.strip()]

    # Предобработка входной заметки
    input_tokens = tokenizer.encode_plus(
        title.strip(),
        text.strip(),
        add_special_tokens=True,
        return_tensors='pt'
    )

    # Получение эмбеддинга входной заметки с помощью DistilBERT
    with torch.no_grad():
        input_embeddings = model(input_tokens['input_ids'])[0][:, 0, :].numpy()

    # Поиск похожих заметок с использованием косинусного сходства
    similar_notes = []
    for note in dataset:
        note_title, note_tags, note_text = note

        # Проверка, есть ли общие теги между входной заметкой и заметкой из датасета
        common_tags = list(set(tags) & set(note_tags))

        # Если есть общие теги, продолжаем обработку
        if common_tags:
            # Предобработка заметки из датасета
            note_tokens = tokenizer.encode_plus(
                note_title,
                note_text,
                add_special_tokens=True,
                return_tensors='pt'
            )

            # Получение эмбеддинга заметки из датасета с помощью DistilBERT
            with torch.no_grad():
                note_embeddings = model(note_tokens['input_ids'])[0][:, 0, :].numpy()

            # Вычисление косинусного сходства 
            similarity = cosine_similarity(input_embeddings, note_embeddings)[0][0]

            # Порог
            if similarity > 0.9:
                similar_notes.append((note_title, note_tags, note_text, similarity))

    if similar_notes:
        return {"similar_notes": similar_notes}
    else:
        return {"message": "Схожих заметок нет"}

def main() -> None:
    """Run application"""
    uvicorn.run("main:app", host="localhost")

if __name__ == "__main__":
    main()
