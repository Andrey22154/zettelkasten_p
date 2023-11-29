import torch
import uvicorn
from threading import Thread
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup

from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel
from typing import List, Optional

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
data = pd.read_csv('C:\\Users\\andre\\Downloads\\notes_project (3).csv')
sentences = data.title.values + data.tags.values + data['main text'].values


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text_pair = self.texts[idx]  # Получаем пару текстов
        label = self.labels[idx]     # Получаем соответствующую метку


        encoding1 = self.tokenizer.encode_plus(
            text_pair[0],  # Первый текст в паре
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )


        encoding2 = self.tokenizer.encode_plus(
            text_pair[1],  # Второй текст в паре
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
    'input_ids_1': encoding1['input_ids'].squeeze(0),
    'attention_mask_1': encoding1['attention_mask'].squeeze(0),
    'input_ids_2': encoding2['input_ids'].squeeze(0),
    'attention_mask_2': encoding2['attention_mask'].squeeze(0),
    'labels': torch.tensor(label, dtype=torch.float)
}


lst_ids = data['linked notes ids'].tolist()
lst_ids = [eval(item) for item in lst_ids]
lst_ids = [[int(element) for element in sublist] for sublist in lst_ids]

max_index = max(max(lst_ids, default=[]), default=0)
label_list = [[0]*len(data) for _ in range(len(data))]

for i, lst in enumerate(lst_ids):
    for index in lst:
        if index < len(data):
            label_list[i][index-1] = 1

# Параметры
MAX_LEN = 320
BATCH_SIZE = 16
EPOCHS = 8

import random

def create_pairs(sentences, label_list):
    print('create_pairs')
    pairs = []
    pair_labels = []

    for idx, (note, label) in enumerate(zip(sentences, label_list)):

        for link_idx in [i for i, l in enumerate(label) if l == 1]:
            pairs.append((note, sentences[link_idx]))
            pair_labels.append(1)


        non_linked_indices = [i for i, l in enumerate(label) if l == 0 and i != idx]
        for _ in range(len(non_linked_indices)):
            random_idx = random.choice(non_linked_indices)
            pairs.append((note, sentences[random_idx]))
            pair_labels.append(-1)

    return pairs, pair_labels

train_texts, train_labels = create_pairs(sentences, label_list)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

train_texts, train_labels = create_pairs(sentences, label_list)


train_texts, test_texts, train_labels, test_labels = train_test_split(
    train_texts,
    train_labels,
    test_size=0.2,
    random_state=42
)

import random

def oversample_positive_pairs(train_texts, train_labels):
    print('oversample_positive_pairs')
    positive_pairs = [pair for pair, label in zip(train_texts, train_labels) if label == 1]
    negative_pairs = [pair for pair, label in zip(train_texts, train_labels) if label == -1]

    # Вычисление, сколько раз нужно повторить положительные пары
    repeat_times = len(negative_pairs) // len(positive_pairs)

    # Oversampling положительных пар
    oversampled_positive_pairs = positive_pairs * repeat_times

    # Объединение oversampled положительных пар с отрицательными парами
    balanced_texts = oversampled_positive_pairs + negative_pairs
    balanced_labels = [1] * len(oversampled_positive_pairs) + [-1] * len(negative_pairs)

    # Перемешивание данных
    combined = list(zip(balanced_texts, balanced_labels))
    random.shuffle(combined)
    balanced_texts, balanced_labels = zip(*combined)

    return balanced_texts, balanced_labels

balanced_train_texts, balanced_train_labels = oversample_positive_pairs(train_texts, train_labels)
balanced_test_texts, balanced_test_labels = oversample_positive_pairs(test_texts, test_labels)

train_dataset = MyDataset(balanced_train_texts, balanced_train_labels, tokenizer, max_len = MAX_LEN)
test_dataset = MyDataset(balanced_test_texts, balanced_test_labels, tokenizer, max_len = MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
print('loader')

import torch.nn as nn

class BertForSimilarity(nn.Module):
    def __init__(self, bert_model):
        super().__init__()
        self.bert = bert_model  # Используйте RoBERTa вместо BERT

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding

similarity_model = BertForSimilarity(model).to(device)

loss_fn = nn.CosineEmbeddingLoss()
optimizer = torch.optim.Adam(similarity_model.parameters(), lr=2e-5, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0.1*total_steps,
                                            num_training_steps = total_steps)

# Функция обучения
def train(model, data_loader, loss_fn, optimizer, device, num_epochs):
    print('train')
    model.train()

    for epoch in range(num_epochs):

        for batch in data_loader:

            input_ids_1 = batch['input_ids_1'].to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            input_ids_2 = batch['input_ids_2'].to(device)
            attention_mask_2 = batch['attention_mask_2'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            embeddings1 = model(input_ids_1, attention_mask_1)
            embeddings2 = model(input_ids_2, attention_mask_2)

            # Вычисление потерь, используя косинусное сходство
            loss = loss_fn(embeddings1, embeddings2, labels)

            loss.backward()
            optimizer.step()
            scheduler.step()


            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

def test(model, data_loader, loss_fn, device, threshold=0.6):
    model.eval()

    total_loss = 0
    total_count = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for batch in data_loader:
            input_ids_1 = batch['input_ids_1'].to(device)
            attention_mask_1 = batch['attention_mask_1'].to(device)
            input_ids_2 = batch['input_ids_2'].to(device)
            attention_mask_2 = batch['attention_mask_2'].to(device)
            labels = batch['labels'].to(device)

            embeddings1 = model(input_ids_1, attention_mask_1)
            embeddings2 = model(input_ids_2, attention_mask_2)

            cos_similarity = F.cosine_similarity(embeddings1, embeddings2)
            probs = (cos_similarity + 1) / 2  # Преобразование в вероятности

            # Преобразование вероятностей в метки на основе порога
            predicted_labels = torch.where(probs >= threshold, 1, -1)


            loss = loss_fn(embeddings1, embeddings2, predicted_labels.float())
            total_loss += loss.item()
            total_count += input_ids_1.size(0)

            predictions.extend(probs.cpu().numpy())
            actuals.extend(labels.cpu().numpy())

    average_loss = total_loss / total_count
    return predictions, actuals, average_loss


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
predictions, actuals, average_loss = test(similarity_model, test_loader, loss_fn, device)

def preprocess_text(text, tokenizer, max_len, device):
    # Токенизация текста
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )
    return inputs.to(device)

def find_similar_notes(input_note, model, dataset, tokenizer, max_len):
    # Обработка входной заметки
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_note_preprocessed = preprocess_text(input_note, tokenizer, max_len, device)

    # Извлечение эмбеддинга для входной заметки
    with torch.no_grad():
        outputs = model(**input_note_preprocessed)
        input_embedding = outputs.last_hidden_state[:, 0, :]

    similarities = []
    for note in dataset:
        # Обработка каждой заметки в датасете
        note_preprocessed = preprocess_text(note, tokenizer, max_len, device)

        # Извлечение эмбеддинга для заметки из датасета
        with torch.no_grad():
            note_outputs = model(**note_preprocessed)
            note_embedding = note_outputs.last_hidden_state[:, 0, :]

        # Вычисление косинусного сходства
        cos_sim = torch.nn.functional.cosine_similarity(input_embedding, note_embedding)
        similarities.append(cos_sim.item())

    # Сортировка и выбор топ-5 схожих заметок
    top_indices = sorted(range(len(similarities)), key=lambda i: similarities[i], reverse=True)[:3]
    top_notes = [dataset[i] for i in top_indices if similarities[i] > 0.9]  # Порог схожести

    return top_notes

# input_note = "Двигатель внутреннего сгоранияДвигатель внутреннего сгорания - это сердце большинства современных автомобилей. Он преобразует химическую энергию топлива в механическую работу, позволяя автомобилю двигаться."
# similar_notes = find_similar_notes(input_note, model, sentences, tokenizer, max_len = MAX_LEN)

# if similar_notes:
#     print("Топ-3 схожих заметок:")
#     for note in similar_notes:
#         print(note)
# else:
#     print("Схожих заметок не найдено.")

class NoteRequest(BaseModel):
    note: str

@app.post("/find_similar")
def find_similar_notes_endpoint(note_request: NoteRequest):
    try:
        input_note = note_request.note
        similar_notes = find_similar_notes(input_note, model, sentences, tokenizer, MAX_LEN)
        return similar_notes
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

import requests

response = requests.post("http://localhost:8000/find_similar", json={"note": "Пример текста заметки"})
if response.status_code == 200:
    print(response.json())
else:
    print("Ошибка:", response.status_code, response.text)

def main() -> None:
    """Run application"""
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)


if __name__ == "__main__":
    main()