import os
import re
import sys
import string

#to find the starting point of fables line no and skip introductory and index part

def find_line_number(file_name, target_string):
    line_number = 0
    blank_lines_count = 0
    with open(file_name, 'r') as file:
        for line in file:
            line_number += 1
            if line.strip() == '':
                blank_lines_count += 1
            else:
                if blank_lines_count == 2 and target_string in line:
                    return line_number
                blank_lines_count = 0
    return -1

#function to extract fables
def extract_fables(file_name):
    collection_folder = 'collection_original'
    if not os.path.exists(collection_folder):
        os.makedirs(collection_folder)
    #call to the function lin number
    linenum = find_line_number(file_name, "Aesop's Fables")
    start_pattern = r"Aesop's Fables" #actucal fabel text starts from "Aesop's Fables"
    start_index = linenum + 3  #+ 3 added beacuse the actual lines are after 3 blank lines

    with open(file_name, 'r') as file:
        lines = file.readlines()

    fables_content = ''.join(lines[start_index - 1:]).strip()
    separated_fables = re.split(r'\n\n\n', fables_content.strip())
    fables = [fable.strip() for fable in separated_fables]
    
    #Lists to store title and texts seperatly
    titles = []
    texts = []

    #Loop to enumerate titel and text
    for j, fable in enumerate(fables, start=1):
        lines = fable.split("\n")
        #If else case because in fabel title and text stored alternately to access then we used if else
        if j % 2 == 1:
            title = lines[0].strip() 
            titles.append(title)
        else:
            text = '\n'.join(lines[0:]).strip()
            texts.append(text)

    for i, (title, text) in enumerate(zip(titles, texts), start=1):
        fable_number = str(i).zfill(2)
        file_name = fable_number + '_' + re.sub(r'[_\s]+', '_', title.lower()).strip(',\'') + '.txt'

        with open(os.path.join(collection_folder, file_name), 'w') as fable_file:
            fable_file.write(text)

    print("Fables extracted successfully.")

#function to remove stopwords
def remove_stopwords(text):
    with open('englishST.txt', 'r') as stopwords_file:
        stopwords = stopwords_file.read().splitlines()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace('\n', ' ')
    words = text.lower().split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

#To remove stopword and stored in new folder
def preprocess_documents(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)

    for file_name in os.listdir(source_folder):
        with open(os.path.join(source_folder, file_name), 'r') as file:
            content = file.read()

        cleaned_content = remove_stopwords(content)

        cleaned_file_name = os.path.splitext(file_name)[0] + '.txt'

        with open(os.path.join(target_folder, cleaned_file_name), 'w') as cleaned_file:
            cleaned_file.write(cleaned_content)

    print("Stop words removed successfully.")

#funstion for linear serach
def linear_search(query, model, documents_folder):
    results = []
    query = query.lower()
#here model is the type of document
    if model == 'original':
        folder = 'collection_original'
    elif model == 'no_stopwords':
        folder = 'collection_no_stopwords'
    else:
        print("Invalid model. Please choose 'original' or 'no_stopwords'.")
        return results

    for file_name in os.listdir(os.path.join(documents_folder, folder)):
        with open(os.path.join(documents_folder, folder, file_name), 'r') as file:
            content = file.read().lower()
            if query in content:
                results.append(file_name)

    return results

if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '-extract-collection':
        file_name = sys.argv[2]
        extract_fables(file_name)
    elif len(sys.argv) == 2 and sys.argv[1] == '-preprocess-documents':
        source_folder = 'collection_original'
        target_folder = 'collection_no_stopwords'
        preprocess_documents(source_folder, target_folder)
    elif len(sys.argv) == 9 and sys.argv[1] == '-model' and sys.argv[2] == 'bool' and sys.argv[3] == '-search-mode' and sys.argv[4] == 'linear' and sys.argv[5] == '-documents' and (sys.argv[6] == 'original' or sys.argv[6] == 'no_stopwords') and sys.argv[7] == '-query':
        model = sys.argv[6]
        query = sys.argv[8]
        results = linear_search(query, model, '.')
        for result in results:
            print(result)
    else:
        print("Invalid command line arguments.")
        print("Usage: python my_ir_system.py -extract-collection aesop10.txt")
        print("Usage: python my_ir_system.py -preprocess-documents")
        print("Usage: python my_ir_system.py -model \"bool\" -search-mode \"linear\" -documents \"original\" -query \"somesearchterm\"")
