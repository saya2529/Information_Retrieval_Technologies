import os
import re
import sys
import string
import time

# Function to read ground_truth.txt file
def read_ground_truth(file_path):
    ground_truth = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            # To avoid commented part of file
            if line and not line.startswith('#'):
                term, ids = line.split(' - ')
                ground_truth[term] = [int(id) for id in ids.split(',')]
    return ground_truth


# Function to find the starting point of fables line number and skip introductory and index part
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


# Function to extract fables
def extract_fables(file_name):
    collection_folder = 'collection_original'
    if not os.path.exists(collection_folder):
        os.makedirs(collection_folder)
    # Call to the function find_line_number
    linenum = find_line_number(file_name, "Aesop's Fables")
    start_pattern = r"Aesop's Fables"  # Actual fable text starts from "Aesop's Fables"
    start_index = linenum + 3  # +3 added because the actual lines are after 3 blank lines

    with open(file_name, 'r') as file:
        lines = file.readlines()

    fables_content = ''.join(lines[start_index - 1:]).strip()
    separated_fables = re.split(r'\n\n\n', fables_content.strip())
    fables = [fable.strip() for fable in separated_fables]

    # Lists to store title and texts separately
    titles = []
    texts = []

    # Loop to enumerate title and text
    for j, fable in enumerate(fables, start=1):
        lines = fable.split("\n")
        # If-else case because in fable title and text are stored alternately
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

    # Preprocess the extracted documents
    source_folder = 'collection_original'
    target_folder = 'collection_no_stopwords'
    # Call this function here to preprocess documents and avoid the use of additional command
    preprocess_documents(source_folder, target_folder)


# ------------------------------------------Stopword remove------------------------------------------------------------
# Function to remove stopwords
def remove_stopwords(text):
    with open('englishST.txt', 'r') as stopwords_file:
        stopwords = stopwords_file.read().splitlines()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace('\n', ' ')
    words = text.lower().split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)


# ------------------------------------------Stemming using porter algorithm------------------------------------------------------------
# Function to apply stemming using the Porter algorithm using porter.txt file
def apply_stemming(word):
    def is_consonant(char):
        vowels = ['a', 'e', 'i', 'o', 'u']
        return char.isalpha() and char.lower() not in vowels

    # Step 1a: Apply rules for plurals
    if word.endswith("sses"):
        word = word[:-2]
    elif word.endswith("ies"):
        word = word[:-2]
    elif word.endswith("s") and not word.endswith("ss"):
        word = word[:-1]

    # Step 1b: Apply rules for specific endings
    if word.endswith("eed"):
        if len(word) > 4:
            word = word[:-1]
    elif word.endswith("ed"):
        if "a" in word[:-2] or "e" in word[:-2] or "i" in word[:-2] or "o" in word[:-2] or "u" in word[:-2]:
            word = word[:-2]
            if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
                word += "e"
            elif (word[-1] == word[-2]) and (word[-1] not in ["l", "s", "z"]):
                word = word[:-1]
        else:
            if len(word) > 4:
                word = word[:-2]
    elif word.endswith("ing"):
        if "a" in word[:-3] or "e" in word[:-3] or "i" in word[:-3] or "o" in word[:-3] or "u" in word[:-3]:
            word = word[:-3]
            if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
                word += "e"
            elif (word[-1] == word[-2]) and (word[-1] not in ["l", "s", "z"]):
                word = word[:-1]
        else:
            if len(word) > 5:
                word = word[:-3]

    # Step 1c: Apply rule for "y" endings
    if word.endswith("y") and len(word) > 2:
        if word[-2] not in ["a", "e", "i", "o", "u"]:
            word = word[:-1] + "i"

    # Step 2: Apply rules for specific endings
    if word.endswith("ational"):
        if len(word) > 8:
            word = word[:-5] + "e"
    elif word.endswith("tional"):
        if len(word) > 7:
            word = word[:-2]
    elif word.endswith("enci"):
        if len(word) > 4:
            word = word[:-1] + "e"
    elif word.endswith("anci"):
        if len(word) > 4:
            word = word[:-1] + "e"
    elif word.endswith("izer"):
        if len(word) > 5:
            word = word[:-1]
    elif word.endswith("abli"):
        if len(word) > 4:
            word = word[:-1] + "e"
    elif word.endswith("alli"):
        if len(word) > 4:
            word = word[:-2]
    elif word.endswith("entli"):
        if len(word) > 5:
            word = word[:-2]
    elif word.endswith("eli"):
        if len(word) > 3:
            word = word[:-2]
    elif word.endswith("ousli"):
        if len(word) > 5:
            word = word[:-2]
    elif word.endswith("ization"):
        if len(word) > 7:
            word = word[:-5] + "e"
    elif word.endswith("ation"):
        if len(word) > 5:
            word = word[:-3] + "e"
    elif word.endswith("ator"):
        if len(word) > 4:
            word = word[:-2]
    elif word.endswith("alism"):
        if len(word) > 5:
            word = word[:-3]
    elif word.endswith("iveness"):
        if len(word) > 7:
            word = word[:-4]
    elif word.endswith("fulness"):
        if len(word) > 7:
            word = word[:-4]
    elif word.endswith("ousness"):
        if len(word) > 7:
            word = word[:-4]
    elif word.endswith("aliti"):
        if len(word) > 4:
            word = word[:-3]
    elif word.endswith("iviti"):
        if len(word) > 4:
            word = word[:-3] + "e"
    elif word.endswith("biliti"):
        if len(word) > 6:
            word = word[:-5] + "le"

    # Step 3: Apply rules for specific endings
    if word.endswith("icate"):
        if len(word) > 5:
            word = word[:-3]
    elif word.endswith("ative"):
        if len(word) > 5:
            word = word[:-5]
    elif word.endswith("alize"):
        if len(word) > 5:
            word = word[:-3]
    elif word.endswith("iciti"):
        if len(word) > 5:
            word = word[:-3]
    elif word.endswith("ical"):
        if len(word) > 4:
            word = word[:-2]
    elif word.endswith("ful"):
        if len(word) > 3:
            word = word[:-3]
    elif word.endswith("ness"):
        if len(word) > 4:
            word = word[:-4]

    # Step 4: Apply rules for specific endings
    if word.endswith("al"):
        if len(word) > 3:
            word = word[:-2]
    elif word.endswith("ance"):
        if len(word) > 4:
            word = word[:-4]
    elif word.endswith("ence"):
        if len(word) > 4:
            word = word[:-4]
    elif word.endswith("er"):
        if len(word) > 2:
            word = word[:-2]
    elif word.endswith("ic"):
        if len(word) > 2:
            word = word[:-2]
    elif word.endswith("able"):
        if len(word) > 4:
            word = word[:-4]
    elif word.endswith("ible"):
        if len(word) > 4:
            word = word[:-4]
    elif word.endswith("ant"):
        if len(word) > 3:
            word = word[:-3]
    elif word.endswith("ement"):
        if len(word) > 5:
            word = word[:-5]
    elif word.endswith("ment"):
        if len(word) > 3:
            word = word[:-4]
    elif word.endswith("ent"):
        if len(word) > 2:
            word = word[:-3]
    elif word.endswith("ion"):
        if len(word) > 3 and word[-4] in ["s", "t"]:
            word = word[:-3]
    elif word.endswith("ou"):
        if len(word) > 2:
            word = word[:-2]
    elif word.endswith("ism"):
        if len(word) > 3:
            word = word[:-3]
    elif word.endswith("ate"):
        if len(word) > 3:
            word = word[:-3]
    elif word.endswith("iti"):
        if len(word) > 3:
            word = word[:-3]
    elif word.endswith("ous"):
        if len(word) > 3:
            word = word[:-3]
    elif word.endswith("ive"):
        if len(word) > 3:
            word = word[:-3]
    elif word.endswith("ize"):
        if len(word) > 3:
            word = word[:-3]

    # Step 5a: Apply rule for "e" endings
    if word.endswith("e") and len(word) > 1:
        if len(word) > 2 or len(word) == 2 and not is_consonant(word[-2]) and is_consonant(word[-3]):
            word = word[:-1]

    # Step 5b: Apply rule for "ll" endings
    if word.endswith("ll") and len(word) > 2:
        word = word[:-1]

    return word


# Function to preprocess the documents
def preprocess_documents(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for file_name in os.listdir(source_folder):
        with open(os.path.join(source_folder, file_name), 'r') as file:
            content = file.read()
        if '--stemming' in sys.argv:
            content = apply_stemming(content)
        else:
            content = remove_stopwords(content)
        target_file_name = os.path.join(target_folder, file_name)
        with open(target_file_name, 'w') as file:
            file.write(content)

    print("Documents preprocessed successfully.")


# ------------------------------------------Search------------------------------------------------------------

## # import os
import re
import sys
import string
import time


# Function to preprocess the documents
def preprocess_documents(source_folder, target_folder):
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    for file_name in os.listdir(source_folder):
        with open(os.path.join(source_folder, file_name), 'r') as file:
            content = file.read()
        if '--stemming' in sys.argv:
            content = apply_stemming(content)
        else:
            content = remove_stopwords(content)
        target_file_name = os.path.join(target_folder, file_name)
        with open(target_file_name, 'w') as file:
            file.write(content)

    print("Documents preprocessed successfully.")


# ------------------------------------------Search------------------------------------------------------------

# Function for linear search
def linear_search(query, model, folder, apply_stemming_flag, ground_truth):
    if apply_stemming_flag:
        query = apply_stemming(query)
    else:
        query = remove_stopwords(query)

    print(f"Model: {model}, Query: {query}")
    print("Search Results:")
    results = []
    start_time = time.time()
    for file_name in os.listdir(folder):
        with open(os.path.join(folder, file_name), 'r') as file:
            content = file.read()
        if apply_stemming_flag:
            content = apply_stemming(content)
        else:
            content = remove_stopwords(content)
        if query.lower() in content.lower():
            results.append(file_name)
    end_time = time.time()
    print_results(results, ground_truth)

    relevant_docs = set()
    for result in results:
        doc_id = int(result.split('_')[0])
        relevant_docs.add(doc_id)

    true_positives = len(relevant_docs.intersection(ground_truth))
    retrieved_docs = len(relevant_docs)

    precision = true_positives / retrieved_docs if retrieved_docs > 0 else 0.0
    recall = true_positives / len(ground_truth) if len(ground_truth) > 0 else 0.0

    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")

    print(f"Search completed in {end_time - start_time} seconds.")
    return results, end_time - start_time, precision, recall


# Main function
def main():
    # Read ground_truth.txt file
    ground_truth = read_ground_truth('ground_truth.txt')

    # Extract fables from the original collection
    file_name = 'collection_original.txt'
    extract_fables(file_name)

    # Read user query from the command line
    query = input("Enter your query: ")

    # Perform linear search on the preprocessed collection without stemming
    model = 'Linear Search'
    folder = 'collection_no_stopwords'
    apply_stemming_flag = False
    results, execution_time, precision, recall = linear_search(query, model, folder, apply_stemming_flag, ground_truth)
    print(f"Search completed in {execution_time:.2f} seconds.")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")



# Main function
def main():
    # Read ground_truth.txt file
    ground_truth = read_ground_truth('ground_truth.txt')

    # Extract fables from the original collection
    file_name = 'collection_original.txt'
    extract_fables(file_name)

    # Read user query from the command line
    query = input("Enter your query: ")

    # Perform linear search on the preprocessed collection without stemming
    model = 'Linear Search'
    folder = 'collection_no_stopwords'
    apply_stemming_flag = False
    results, execution_time = linear_search(query, model, folder, apply_stemming_flag, ground_truth)
    print(f"Search completed in {execution_time:.2f} seconds.")




# Main function
def main():
    # Read ground_truth.txt file
    ground_truth = read_ground_truth('ground_truth.txt')

    # Extract fables from the original collection
    file_name = 'collection_original.txt'
    extract_fables(file_name)

    # Read user query from the command line
    query = input("Enter your query: ")

    # Perform linear search on the preprocessed collection without stemming
    model = 'Linear Search'
    folder = 'collection_no_stopwords'
    apply_stemming_flag = False
    results, execution_time = linear_search(query, model, folder, apply_stemming_flag, ground_truth)
    print(f"Search completed in {execution_time:.2f} seconds.")






# Function to print search results
def print_results(results, ground_truth):
    for result in results:
        fable_number = result[:2]
        if int(fable_number) in ground_truth:
            relevance = 'Relevant'
        else:
            relevance = 'Not Relevant'
        print(f"{result}: {relevance}")


# Main function
def main():
    # Read ground_truth.txt file
    ground_truth = read_ground_truth('ground_truth.txt')

    # Extract fables from the original collection
    file_name = 'collection_original.txt'
    extract_fables(file_name)

    # Read user query from the command line
    query = input("Enter your query: ")

    # Perform linear search on the preprocessed collection without stemming
    model = 'Linear Search'
    folder = 'collection_no_stopwords'
    apply_stemming_flag = False
    linear_search(query, model, folder, apply_stemming_flag, ground_truth)

# Read the ground_truth.txt file
ground_truth = read_ground_truth("ground_truth.txt")

# Extract fables from the given file
file_name = "aesopa10.txt"
extract_fables(file_name)

# Define the model and folder paths
models = ["linear", "inverted"]
folders = ["collection_no_stopwords", "collection_stemming"]

# Query examples
queries = ["fox", "lion", "rabbit", "fox & lion", "fox | lion", "-fox", "-lion", "fox & -lion", "fox | -lion"]

# Perform search for each model and folder combination
for model in models:
    for folder in folders:
        print(f"\n==============================\nModel: {model}, Folder: {folder}\n==============================")
        if model == "linear":
            for query in queries:
                results, execution_time, precision, recall = linear_search(query, model, folder, folder == "collection_stemming", ground_truth)
                print(f"Query: {query}")
                print(f"Results: {results}")
                print(f"Execution Time: {execution_time:.2f} ms")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                print()
        elif model == "inverted":
            for query in queries:
                results, execution_time, precision, recall = inverted_list_search(query, model, folder, folder == "collection_stemming", ground_truth)
                print(f"Query: {query}")
                print(f"Results: {results}")
                print(f"Execution Time: {execution_time:.2f} ms")
                print(f"Precision: {precision}")
                print(f"Recall: {recall}")
                print()


