import os
import re
import sys
import string
import time
from collections import Counter
import math


#function toread ground_truth.txt file
def read_ground_truth(file_path):
    ground_truth = {}
    with open(file_path, 'r') as file:
        for line in file:
            line = line.strip()
            #to avoid commented part of file
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
    # call this function here to preprocess documents and avoid us of additional command
    preprocess_documents(source_folder, target_folder)

#------------------------------------------Stopword remove------------------------------------------------------------
# Function to remove stopwords
def remove_stopwords(text):
    with open('englishST.txt', 'r') as stopwords_file:
        stopwords = stopwords_file.read().splitlines()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.replace('\n', ' ')
    words = text.lower().split()
    filtered_words = [word for word in words if word not in stopwords]
    return ' '.join(filtered_words)

#------------------------------------------Stemming using porter algorithm------------------------------------------------------------
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

#------------------------------------------Search------------------------------------------------------------

# Function for linear search
def linear_search(query, model, folder, apply_stemming_flag,ground_truth):
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

        if query.lower() in content.lower():
            results.append(file_name)
    end_time = time.time()
    execution_time = (end_time - start_time) * 1000

    relevant_docs = set()

    for result in results:
        doc_id = int(result.split('_')[0])
        relevant_docs.add(doc_id)

    precision = 0.0  # Initialize with default value
    recall = 0.0  # Initialize with default value

    if query in ground_truth:
        relevant_docs_ground_truth = set(ground_truth[query])
        true_positives = len(relevant_docs.intersection(relevant_docs_ground_truth))
        retrieved_docs = len(relevant_docs)

        if retrieved_docs > 0:
            precision = true_positives / retrieved_docs

        if len(relevant_docs_ground_truth) > 0:
            recall = true_positives / len(relevant_docs_ground_truth)
        precision_str = f"P={precision:.2f}"
        recall_str = f"R={recall:.2f}"
    else:
      precision_str="P=?"
      recall_str="R=?"
    return results, execution_time, precision_str, recall_str




# Function for inverted list search
def inverted_list_search(query, model, folder, apply_stemming_flag,ground_truth):
    if apply_stemming_flag:
        query = apply_stemming(query)
    else:
        query = remove_stopwords(query)

    print(f"Model: {model}, Query: {query}")
    print("Search Results:")

    inverted_index = {}
    start_time = time.time()
    for file_name in os.listdir(folder):
        with open(os.path.join(folder, file_name), 'r') as file:
            content = file.read()
        words = content.lower().split()
        translator = str.maketrans("", "", string.punctuation)  # Translator to remove punctuation marks
        for word in words:
            word = word.translate(translator)  # Remove punctuation marks from the word
            if word:
                if word not in inverted_index:
                    inverted_index[word] = []
                inverted_index[word].append(file_name)

    # Process the query terms
    results = None
    operators = ['&', '|', '-']
    query_terms = []
    query_operator = None

    for operator in operators:
        if operator in query:
            query_terms = query.split(operator)
            query_operator = operator
            break

    start_time = time.time()
#for conjuction
    if query_operator == '&':
        term1 = query_terms[0].strip()
        term2 = query_terms[1].strip()
        results_term1 = set(inverted_index.get(term1, []))
        results_term2 = set(inverted_index.get(term2, []))
        results = list(results_term1.intersection(results_term2))
#for disjuction
    elif query_operator == '|':
        term1 = query_terms[0].strip()
        term2 = query_terms[1].strip()
        results = list(set(inverted_index.get(term1, [])) | set(inverted_index.get(term2, [])))
#for negation
    elif query_operator == '-':
        negation_term = query_terms[0].strip()[1:]
        results = list(set(os.listdir(folder)) - set(inverted_index.get(negation_term, [])))

    precision = recall = 0.0
    relevant_docs = set()

    for result in results:
        doc_id = int(result.split('_')[0])
        relevant_docs.add(doc_id)

    if query_terms:
        relevant_docs_ground_truth = set()
        for term in query_terms:
            if term in ground_truth:
                relevant_docs_ground_truth.update(ground_truth[term])

        true_positives = len(relevant_docs.intersection(relevant_docs_ground_truth))
        retrieved_docs = len(relevant_docs)

        if retrieved_docs > 0:
            precision = true_positives / retrieved_docs

        if len(relevant_docs_ground_truth) > 0:
            recall = true_positives / len(relevant_docs_ground_truth)
       
    end_time = time.time()
    precision_str = f"P={precision:.2f}" if precision > 0 else "P=?"
    recall_str = f"R={recall:.2f}" if recall > 0 else "R=?"
    execution_time = (end_time - start_time) * 1000
    return results,execution_time,precision_str,recall_str

#------------------------------Function for inverted list search with Vector Space Model--------------------------------
def inverted_list_search_vsm(query, folder, apply_stemming_flag, ground_truth):
    if apply_stemming_flag:
        query = apply_stemming(query)
    else:
        query = remove_stopwords(query)

    print(f"Model: vector, Query: {query}")
    print("Search Results:")

    inverted_index = {}
    document_lengths = {}
    num_documents = 0

    start_time = time.time()

    # Compute inverted index and document lengths
    for file_name in os.listdir(folder):
        with open(os.path.join(folder, file_name), 'r') as file:
            content = file.read()

        words = content.lower().split()
        translator = str.maketrans("", "", string.punctuation)  # Translator to remove punctuation marks

        document_id = int(file_name.split('_')[0])

        document_lengths[document_id] = 0  # Initialize document length

        for word in words:
            word = word.translate(translator)  # Remove punctuation marks from the word
            if word:
                if word not in inverted_index:
                    inverted_index[word] = []
                if document_id not in inverted_index[word]:
                    inverted_index[word].append(document_id)
                    document_lengths[document_id] += 1  # Increment document length
        num_documents += 1  # Increment total number of documents

    # Compute tf-idf weights for the query terms
    query_terms = query.split()
    query_term_freqs = Counter(query_terms)
    query_vector = {}

    for term, freq in query_term_freqs.items():
        tf = 1 + math.log10(freq)  # Apply tf weighting
        df = len(inverted_index.get(term, []))
        idf = math.log10(num_documents / df) if df > 0 else 0  # Compute idf
        tf_idf = tf * idf  # Compute tf-idf weight
        query_vector[term] = tf_idf

    # Compute the query vector length
    query_vector_length = math.sqrt(sum(math.pow(weight, 2) for weight in query_vector.values()))

    # Compute document scores
    document_scores = {}

    for term, weight in query_vector.items():
        if term in inverted_index:
            postings = inverted_index[term]
            for document_id in postings:
                document_length = document_lengths[document_id]
                if document_length > 0:  # Check if document length is greater than zero
                    tf_idf = weight * (1 + math.log10(document_length))  # Apply tf weighting on document length
                    if document_id not in document_scores:
                        document_scores[document_id] = 0
                    document_scores[document_id] += tf_idf

    # Normalize document scores by document vector lengths
    for document_id, score in document_scores.items():
        document_vector_length = math.sqrt(document_lengths[document_id])
        document_scores[document_id] /= (query_vector_length * document_vector_length)

    # Sort documents based on scores
    results = sorted(document_scores.keys(), key=lambda doc_id: document_scores[doc_id], reverse=True)

    precision = recall = 0.0
    relevant_docs = set()

    for result in results:
        doc_id = result
        relevant_docs.add(doc_id)

    if query_terms:
        relevant_docs_ground_truth = set()
        for term in query_terms:
            if term in ground_truth:
                relevant_docs_ground_truth.update(ground_truth[term])

        true_positives = len(relevant_docs.intersection(relevant_docs_ground_truth))
        retrieved_docs = len(relevant_docs)

        if retrieved_docs > 0:
            precision = true_positives / retrieved_docs

        if len(relevant_docs_ground_truth) > 0:
            recall = true_positives / len(relevant_docs_ground_truth)

    end_time = time.time()

    precision_str = f"P={precision:.2f}" if precision > 0 else "P=?"
    recall_str = f"R={recall:.2f}" if recall > 0 else "R=?"
    execution_time = (end_time - start_time) * 1000
    # Sort documents based on scores

    return results, execution_time, precision_str, recall_str


if __name__ == '__main__':
    if len(sys.argv) == 3 and sys.argv[1] == '--extract-collection':
        file_name = sys.argv[2]
        extract_fables(file_name)
    elif (
            len(sys.argv) == 10
            and sys.argv[1] == '--model'
            and sys.argv[2] == 'bool'
            and sys.argv[3] == '--search-mode'
            and (sys.argv[4] == 'linear' or sys.argv[4] == 'inverted')
            and sys.argv[5] == '--documents'
            and (sys.argv[6] == 'original' or sys.argv[6] == 'no_stopwords')
            and sys.argv[7] == '--stemming'
            and sys.argv[8] == '--query'
    ):
        model = sys.argv[6]
        apply_stemming_flag = True
        query = sys.argv[9]
        if model == "original":
            folder = "collection_original"
        else:
            folder = "collection_no_stopwords"
        ground_truth = read_ground_truth("ground_truth.txt")

        if sys.argv[4] == 'linear':
            results, execution_time, precision_str, recall_str = linear_search(query, model, folder,
                                                                              apply_stemming_flag, ground_truth)
        elif sys.argv[4] == 'inverted':
            results, execution_time, precision_str, recall_str = inverted_list_search(query, model, folder,
                                                                                     apply_stemming_flag, ground_truth)
        elif sys.argv[4] == 'vector':
            results, execution_time, precision_str, recall_str = inverted_list_search_vsm(query, folder,
                                                                                          apply_stemming_flag,
                                                                                          ground_truth)
        else:
            print("Invalid search mode.")
            sys.exit(1)

        for result in results:
            print(result)
        print(f"T={execution_time:.2f}ms, {precision_str}, {recall_str}")
    elif (
            len(sys.argv) == 8
            and sys.argv[1] == '--model'
            and sys.argv[2] == 'vector'
            and sys.argv[3] == '--documents'
            and (sys.argv[4] == 'original' or sys.argv[6] == 'no_stopwords')
            and sys.argv[5] == '--stemming'
            and sys.argv[6] == '--query'
    ):
        model = sys.argv[4]
        apply_stemming_flag = True
        query = sys.argv[7]
        if model == "original":
            folder = "collection_original"
        else:
            folder = "collection_no_stopwords"
        ground_truth = read_ground_truth("ground_truth.txt")

        if sys.argv[2] == 'vector':
            results, execution_time, precision_str, recall_str = inverted_list_search_vsm(query, folder,
                                                                                          apply_stemming_flag,
                                                                                          ground_truth)
        else:
            print("Invalid search mode.")
            sys.exit(1)

        print(f"T={execution_time:.2f}ms, {precision_str}, {recall_str}")
    else:
        print("Invalid command line arguments.")
        print("Usage: python my_ir_system.py --extract-collection aesop10.txt")
        print(
            "Usage: python my_ir_system.py --model \"bool\" --search-mode \"linear\" --documents \"original\" --stemming --query \"somesearchterm\"")
        print(
            "Usage: python my_ir_system.py --model \"bool\" --search-mode \"inverted\" --documents \"original\" --stemming --query \"somesearchterm(fox|wolf\"")