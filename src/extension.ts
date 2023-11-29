import * as vscode from 'vscode';
import axios from 'axios';

export function activate(context: vscode.ExtensionContext) {
    console.log('simNotes extension is now active');

    // Создание канала вывода
    const outputChannel = vscode.window.createOutputChannel("Similar Notes");

    // Command for 'Hello World'
    let helloWorldDisposable = vscode.commands.registerCommand('simNotes.helloWorld', () => {
        vscode.window.showInformationMessage('Hello World from simNotes!');
    });
    context.subscriptions.push(helloWorldDisposable);

    // Command for 'Find Similar Notes'
    let findSimilarDisposable = vscode.commands.registerCommand('simNotes.findSimilarNotes', async () => {
        // Get the text from the active editor
        const editor = vscode.window.activeTextEditor;
        if (editor) {
            const document = editor.document;
            const noteText = document.getText(editor.selection);

            // Send request to your FastAPI server
            try {
                const response = await axios.post('http://localhost:8000/find_similar', { note: noteText });
                const similarNotes = response.data;

                // Очистка и вывод результатов в канал вывода
                outputChannel.clear();
                outputChannel.appendLine(`Found ${similarNotes.length} similar notes:`);
                similarNotes.forEach((note: string) => {
                    outputChannel.appendLine(note);
                });
                outputChannel.show();
            } catch (error) {
                vscode.window.showErrorMessage('Error fetching similar notes');
                console.error(error);
            }
        }
    });
    context.subscriptions.push(findSimilarDisposable);
}

export function deactivate() {}



