import os
import re
import sys
import string
import time
from math import log10, sqrt


# Function to read the ground_truth.txt file
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
            text = file.read()

        # Remove stopwords
        text = remove_stopwords(text)

        # Apply stemming
        words = text.split()
        stemmed_words = [apply_stemming(word) for word in words]
        processed_text = ' '.join(stemmed_words)

        # Write processed text to target folder
        with open(os.path.join(target_folder, file_name), 'w') as file:
            file.write(processed_text)

    print("Documents preprocessed successfully.")


# Function to calculate the TF-IDF vector
def calculate_tf_idf_vector(documents_folder):
    document_vectors = {}
    document_frequencies = {}
    num_documents = 0

    # Calculate document frequencies
    for file_name in os.listdir(documents_folder):
        with open(os.path.join(documents_folder, file_name), 'r') as file:
            document = file.read()
        words = document.split()
        unique_words = set(words)
        for word in unique_words:
            if word in document_frequencies:
                document_frequencies[word] += 1
            else:
                document_frequencies[word] = 1

        num_documents += 1

    # Calculate TF-IDF vectors
    for file_name in os.listdir(documents_folder):
        with open(os.path.join(documents_folder, file_name), 'r') as file:
            document = file.read()
        words = document.split()
        word_counts = {}
        max_word_count = 0
        for word in words:
            if word in word_counts:
                word_counts[word] += 1
            else:
                word_counts[word] = 1
            max_word_count = max(max_word_count, word_counts[word])

        tf_idf_vector = {}
        for word in words:
            tf = word_counts[word] / max_word_count
            idf = log10(num_documents / document_frequencies[word])
            tf_idf = tf * idf
            tf_idf_vector[word] = tf_idf

        document_vectors[file_name] = tf_idf_vector

    return document_vectors


# Function to calculate the cosine similarity between two vectors
def calculate_cosine_similarity(vector1, vector2):
    dot_product = 0
    magnitude1 = 0
    magnitude2 = 0
    for word in vector1:
        dot_product += vector1[word] * vector2.get(word, 0)
        magnitude1 += vector1[word] ** 2
    for word in vector2:
        magnitude2 += vector2[word] ** 2
    magnitude1 = sqrt(magnitude1)
    magnitude2 = sqrt(magnitude2)
    if magnitude1 != 0 and magnitude2 != 0:
        similarity = dot_product / (magnitude1 * magnitude2)
    else:
        similarity = 0
    return similarity


# Function to perform linear search using vector space model
def linear_search(query, model, folder, stemming, ground_truth):
    if stemming:
        query = remove_stopwords(query)
        query_words = query.split()
        query_words = [apply_stemming(word) for word in query_words]
        query = ' '.join(query_words)
        folder += '_stemming'
    else:
        query = remove_stopwords(query)

    start_time = time.time()

    query_vector = {}
    words = query.split()
    word_counts = {}
    max_word_count = 0
    for word in words:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1
        max_word_count = max(max_word_count, word_counts[word])

    for word in words:
        tf = word_counts[word] / max_word_count
        idf = log10(len(os.listdir(folder)) / model.get(word, 0))
        tf_idf = tf * idf
        query_vector[word] = tf_idf

    results = []
    for file_name in os.listdir(folder):
        with open(os.path.join(folder, file_name), 'r') as file:
            document = file.read()
        similarity = calculate_cosine_similarity(query_vector, model[file_name])
        results.append((file_name, similarity))

    results.sort(key=lambda x: x[1], reverse=True)
    execution_time = time.time() - start_time

    # Calculate precision and recall
    retrieved_docs = [result[0] for result in results]
    relevant_docs = ground_truth.get(query, [])
    tp = len(set(retrieved_docs).intersection(relevant_docs))
    precision = tp / len(retrieved_docs) if retrieved_docs else 0
    recall = tp / len(relevant_docs) if relevant_docs else 0

    return results, execution_time, precision, recall


# Function to perform search using vector space model
def vector_space_model_search(model, folder, stemming, ground_truth):
    while True:
        query = input("Enter query (or 'exit' to quit): ")
        if query == 'exit':
            break

        results, execution_time, precision, recall = linear_search(query, model, folder, stemming, ground_truth)

        print(f"\nSearch Results for '{query}':")
        print("-------------------------------------------------------")
        print(f"Search took {execution_time:.6f} seconds.")
        print(f"Precision: {precision:.2%}")
        print(f"Recall: {recall:.2%}")
        print("-------------------------------------------------------")
        for i, result in enumerate(results, start=1):
            print(f"{i}. Document: {result[0]}")
            print(f"   Similarity: {result[1]:.6f}")
            print("-------------------------------------------------------")


# Main function
if __name__ == '__main__':
    # Extract fables from the file
    extract_fables('aesop.txt')

    # Preprocess the documents
    preprocess_documents('collection_original', 'collection_preprocessed')

    # Calculate the TF-IDF vector
    tf_idf_model = calculate_tf_idf_vector('collection_preprocessed')

    # Read the ground_truth.txt file
    ground_truth = read_ground_truth('ground_truth.txt')

    # Perform search using vector space model
    vector_space_model_search(tf_idf_model, 'collection_preprocessed', False, ground_truth)
