# simNotes
simNotes is a Visual Studio Code extension that analyzes note text and suggests a list of similar notes. Utilizing machine learning and NLP (Natural Language Processing), the extension helps organize and link your notes, making it easier to find relevant information.

## Features
Note Text Analysis: Automatically analyzes the content of notes.
Similar Notes Search: Finds and suggests notes that are similar in content.
Easy Integration: Operates directly within the VS Code environment, simplifying the note management process.
Requirements
To use the extension, you must have Visual Studio Code version 1.84.0 or higher installed.

## Installation and Setup
Install the extension via the Visual Studio Code Marketplace.
Ensure Python and necessary dependencies for the machine learning model are installed on your system.
Run a local FastAPI server to handle requests from the extension.
Usage
You can create your dataset in the same format and convert it to csv using the code in the data->convert_to_csv folder.
Then train model_bert from the model folder on this data and run the main.py file on your local computer on the saved weights.

## If you want to work only with FastAPI
Launch the local FastAPI server by running the main.py code on your local computer.
Send a request with the text of the note to the FastAPI server.
View and analyze the results in the server response.

## Using in Visual Studio Code
Launch the local FastAPI server by running the main.py code on your local computer.
Run the code in out->test->extension.ts and open a note in Visual Studio Code.
Enter the text of the note.
Use the Find Similar Notes command to search for similar notes.
The results will be displayed in a special output window.
