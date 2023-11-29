import torch

# If there's a GPU available...
if torch.cuda.is_available():

    # Tell PyTorch to use the GPU.
    device = torch.device("cuda")

    print('There are %d GPU(s) available.' % torch.cuda.device_count())

    print('We will use the GPU:', torch.cuda.get_device_name(0))

# If not...
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

import torch
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from transformers import get_linear_schedule_with_warmup
import pandas as pd

# Класс датасета
import torch
from transformers import BertTokenizer

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


data = pd.read_csv('path')

sentences = data.title.values + data.tags.values + data['main text'].values
lst_ids = data['linked notes ids'].tolist()
lst_ids = [eval(item) for item in lst_ids]
lst_ids = [[int(element) for element in sublist] for sublist in lst_ids]

max_index = max(max(lst_ids, default=[]), default=0)
label_list = [[0]*len(data) for _ in range(len(data))]

for i, lst in enumerate(lst_ids):
    for index in lst:
        if index < len(data):
            label_list[i][index-1] = 1

from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
token_lengths = []

for text in sentences:
    tokens = tokenizer.encode(text, add_special_tokens=True)
    token_lengths.append(len(tokens))

avg_length = sum(token_lengths) / len(token_lengths)
max_length = max(token_lengths)

print(f'Average token length: {avg_length}')
print(f'Maximum token length: {max_length}')

# Параметры
MAX_LEN = 320
BATCH_SIZE = 16
EPOCHS = 8

import random

def create_pairs(sentences, label_list):
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

from sklearn.model_selection import train_test_split


train_texts, train_labels = create_pairs(sentences, label_list)


train_texts, test_texts, train_labels, test_labels = train_test_split(
    train_texts,
    train_labels,
    test_size=0.2,
    random_state=42
)

import random

def oversample_positive_pairs(train_texts, train_labels):
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

class BertForSimilarity(nn.Module):
    def __init__(self, roberta_model):
        super().__init__()
        self.roberta = roberta_model  # Используйте RoBERTa вместо BERT

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids, attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        return cls_embedding
    
similarity_model = BertForSimilarity(model).to(device)

loss_fn = nn.CosineEmbeddingLoss()
optimizer = torch.optim.Adam(similarity_model.parameters(), lr=2e-5, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = 0.1*total_steps,
                                            num_training_steps = total_steps)

def train(model, data_loader, loss_fn, optimizer, device, num_epochs):
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


train(similarity_model, train_loader, loss_fn, optimizer, device, EPOCHS)

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
print("Среднее значение потерь на тестовом наборе:", average_loss)

from sklearn.metrics import precision_score, recall_score

def calculate_precision_recall_sklearn(predictions, actuals, threshold=0.6):

    predicted_labels = [1 if p >= threshold else 0 for p in predictions]


    actuals_transformed = [1 if a == 1 else 0 for a in actuals]


    precision = precision_score(actuals_transformed, predicted_labels)
    recall = recall_score(actuals_transformed, predicted_labels)

    return precision, recall


precision, recall = calculate_precision_recall_sklearn(predictions, actuals)
print(f"Precision: {precision}")
print(f"Recall: {recall}")

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

input_note = "Двигатель внутреннего сгоранияДвигатель внутреннего сгорания - это сердце большинства современных автомобилей. Он преобразует химическую энергию топлива в механическую работу, позволяя автомобилю двигаться."
similar_notes = find_similar_notes(input_note, model, sentences, tokenizer, max_len = MAX_LEN)

if similar_notes:
    print("Топ-3 схожих заметок:")
    for note in similar_notes:
        print(note)
else:
    print("Схожих заметок не найдено.")

model.save_pretrained('path')

from google.colab import files
files.download('path')
files.download('path')


