import re
import pandas as pd

def process_notes_file(input_file):
    '''
    This function processes a notes file and extracts relevant information from it.
    It takes an input file path as a parameter.
    '''

    # Read the contents of the input file
    with open(input_file, "r", encoding="utf-8") as file:
        document_content = file.read()

    # Split the document content into individual notes
    document_content = document_content.split('\n')
    document_content = [i[3:] for i in document_content]
    titles = []
    tags = []
    notes = []
    sim = []

    # Process each note
    for note in document_content:
        # Extract the title from the note
        title = []
        for i in note:
            if i != '#':
                title.append(i)
            else:
                break

        # Extract the tags from the note
        tag = ''.join(re.findall(r"#\w+", note)) + '.'
        tag = tag.replace("#", " #")[1:]
        tags.append(tag)

        # Remove the title and tags from the note
        title = ''.join(title)
        titles.append(title)
        note_with_sim = note.replace(title + tag, '')[1:]

        # Extract the linked note IDs from the note
        last_note = re.findall('\\[(.*?)\\]$', note_with_sim)
        if last_note:
            l = [i.split(',') for i in last_note]
            l = [item.strip() for sublist in l for item in sublist]
            l = [int(i) for i in l if i.isdigit()]
            last_note = str(l)
        else:
            last_note = ''
        sim.append(last_note)

        # Remove the linked note IDs from the note
        number_sim = note_with_sim.replace(last_note, '')
        notes.append(number_sim)

    # Create a DataFrame with the extracted information
    data = pd.DataFrame({'title': titles, 'tags': tags, 'main text': notes, 'linked notes ids': sim})
    data.index = range(1, len(data) + 1)
    lists = [eval(s) if s else [] for s in data['linked notes ids']]
    data['linked notes ids'] = lists
    data.to_csv('path', index=False, encoding='utf-8')