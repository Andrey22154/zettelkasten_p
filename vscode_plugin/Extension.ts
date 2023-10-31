import * as vscode from 'vscode';
import { DistilBertTokenizer, DistilBertModel } from 'transformers';
import { Note } from './Note';

const tokenizer = DistilBertTokenizer.fromPretrained('distilbert-base-uncased');
const model = DistilBertModel.fromPretrained('distilbert-base-uncased');

interface SimilarNote {
  title: string;
  tags: string[];
  text: string;
  similarity: number;
}

const dataset: Note[] = [];

function saveNoteCommand() {
  const note = vscode.window.showInputBox({ prompt: 'Введите заметку' });
  note.then((value) => {
    if (value) {
      const result = saveNote(value);
      vscode.window.showInformationMessage(result);
    }
  });
}

function findSimilarNotesCommand() {
  const note = vscode.window.showInputBox({ prompt: 'Введите заметку' });
  note.then((value) => {
    if (value) {
      const result = findSimilarNotes(value);
      if (Array.isArray(result)) {
        const similarNotes = result.map((note) => `${note.title}: ${note.similarity}`);
        vscode.window.showInformationMessage(similarNotes.join('\n'));
      } else {
        vscode.window.showInformationMessage(result);
      }
    }
  });
}

function saveNote(note: string): string {
  const [title, tags, text] = note.split(".");

  const tagList = tags.split("#").map(tag => tag.trim()).filter(tag => tag !== "");

  dataset.push({ title: title.trim(), tags: tagList, text: text.trim() });

  return "Заметка сохранена успешно";
}

// Define the function to calculate cosine similarity
function cosineSimilarity(A: number[], B: number[]): number {
  // Initialize the sums
  let sumAiBi = 0, sumAiAi = 0, sumBiBi = 0;

  // Iterate over the elements of vectors A and B
  for (let i = 0; i < A.length; i++) {
      // Calculate the sum of Ai*Bi
      sumAiBi += A[i] * B[i];
      // Calculate the sum of Ai*Ai
      sumAiAi += A[i] * A[i];
      // Calculate the sum of Bi*Bi
      sumBiBi += B[i] * B[i];
  }

  // Calculate and return the cosine similarity
  return 1.0 - sumAiBi / Math.sqrt(sumAiAi * sumBiBi);
}

function findSimilarNotes(note: string): SimilarNote[] | string {
  const [title, tags, text] = note.split(".");

  const tagList = tags.split("#").map(tag => tag.trim()).filter(tag => tag !== "");

  const inputTokens = tokenizer.encodePlus(
    title.trim(),
    text.trim(),
    true,
    'pt'
  );

  const inputEmbeddings = model(inputTokens['input_ids'])[0].slice([null, 0, null]).numpy();

  const similarNotes: SimilarNote[] = [];
  for (const note of dataset) {
    const { title: noteTitle, tags: noteTags, text: noteText } = note;

    const commonTags = tagList.filter(tag => noteTags.includes(tag));

    if (commonTags.length > 0) {
      const noteTokens = tokenizer.encodePlus(
        noteTitle,
        noteText,
        true,
        'pt'
      );

      const noteEmbeddings = model(noteTokens['input_ids'])[0].slice([null, 0, null]).numpy();

      const similarity = cosineSimilarity(inputEmbeddings, noteEmbeddings)[0][0];

      if (similarity > 0.9) {
        similarNotes.push({ title: noteTitle, tags: noteTags, text: noteText, similarity });
      }
    }
  }

  if (similarNotes.length > 0) {
    return similarNotes;
  } else {
    return "Схожих заметок нет";
  }
}

export function activate(context: vscode.ExtensionContext) {
  context.subscriptions.push(vscode.commands.registerCommand('extension.saveNote', saveNoteCommand));
  context.subscriptions.push(vscode.commands.registerCommand('extension.findSimilarNotes', findSimilarNotesCommand));
}

export function deactivate() {}

