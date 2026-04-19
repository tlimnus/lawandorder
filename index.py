import sys
import getopt
import nltk
#Use Python CSV Module to read the dataset
import csv
import math

import re

from nltk.stem import PorterStemmer

# Use maxInt to set the maximum field size limit for CSV reader, this is needed because some cases have very long content
maxInt = sys.maxsize

while True:
    # decrease the maxInt value by factor 10 
    # as long as the OverflowError occurs.
    try:
        csv.field_size_limit(maxInt)
        break
    except OverflowError:
        maxInt = int(maxInt/10)

def usage():
    print("usage: " + sys.argv[0] + " -i directory-of-documents -d dictionary-file -p postings-file")

# This function reads the CSV file and yields one case at a time as a dictionary with keys: document_id, title, content, date_posted, court
def iter_cases(input_directory):

    # Read the CSV File, it is assigned to input_directory variable in the main function
    with open(input_directory, "r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)

        # Use yield so that we don't load all cases into memory at once, we can iterate one by one
        # yield returns a generator object, which is an iterator
        for row in reader:
            yield {
                "document_id": row["document_id"],
                "title": row["title"],
                "content": row["content"],
                "date_posted": row["date_posted"],
                "court": row["court"],
            }
            
# Function to Normalize Tokens by extracting alphabetic subwords,
# then lowercasing and stemming them.
def normalize_tokens(tokens, porter):
    normalized = []

    for token in tokens:
        # Extract alphabetic chunks only.
        # Examples:
        # "damages.171" -> ["damages"]
        # "damages-art" -> ["damages", "art"]
        # "(Damages)" -> ["Damages"]
        # "s.33" -> ["s"]
        subwords = re.findall(r"[A-Za-z]+", token)

        for subword in subwords:
            subword = subword.lower()
            normalized.append(porter.stem(subword))

    return normalized

# Function to Add Term to the Index Dictionary, used for both court indexing
def add_term(postings_dict, term, doc_id):
    # If Term is not in posting_dict, initialize term and doc_id
    if term not in postings_dict:
        postings_dict[term] = {doc_id: 1}
    # elif the term is already initialize but doc_id is not inside, initialize doc_id
    elif doc_id not in postings_dict[term]:
        postings_dict[term][doc_id] = 1
    # if term and doc_id is inside, then can increment it
    else:
        postings_dict[term][doc_id] += 1

def add_term_position(postings_dict, term, position, doc_id):
    # If term is not in posting_dict, initialise
    if term not in postings_dict:
        postings_dict[term] = {}
    if doc_id not in postings_dict[term]:
        postings_dict[term][doc_id] = []
    # Append the position of the term
    postings_dict[term][doc_id].append(position)
        
# Function to build index for Content and Title of the Document
def index_content_title_text(text, doc_id, postings_dict, porter):
    # Use NLTK Word Tokenizer to tokenize the words in the text
    words = nltk.word_tokenize(text)
    # Normalize the Tokens by Lowercasing and Stemming using Porter Stemmer
    normalized_words = normalize_tokens(words, porter)
    # Add Each Normalized Word to the Index Dictionary with the Document ID and Term Frequency
    for position, word in enumerate(normalized_words):
        add_term_position(postings_dict, word, position, doc_id)
        
# Function to build index for Content of the Document
def index_court(court_name, doc_id, postings_dict):
    # Preserve Court Name Exactly because they have special semantic meaning
    add_term(postings_dict, court_name, doc_id)

# Function to do Document_Length_Calculation for LNC Weighting Scheme
# Logarithm for Term Frequency (TF)
# Square Root for Document Length Normalization

def document_length_calculation(index_dictionary):
    document_lengths = {}
    for term in index_dictionary:
        for doc_id, positional_indice in index_dictionary[term].items():
            # Calculate the Term Frequency (TF) term for LNC weighting scheme
            tf_weight = 1 + math.log10(len(positional_indice))
            if doc_id not in document_lengths:
                document_lengths[doc_id] = 0
            
            document_lengths[doc_id] += tf_weight ** 2
            
    for doc_id in document_lengths:
        document_lengths[doc_id] = math.sqrt(document_lengths[doc_id])
        
    return document_lengths

# Function to write positional index to dictionary file
def write_positional_index_to_dict(index_dictionary, dict_file, postings_file):
    # Sort the terms alphabetically
    for term in sorted(index_dictionary.keys()):
        # Get the offset of the postings_file
        offset = postings_file.tell()
        number_of_documents = len(index_dictionary[term])
        dict_file.write(f"{term} {number_of_documents} {offset}\n")

        postings_parts = []
        # Sort based on document id
        for doc_id, positions in sorted(index_dictionary[term].items()):
            # join the positions with comma and append to postings_parts using f string
            positions_str = ",".join(map(str, positions))
            postings_parts.append(f"{doc_id}:{positions_str}")

        postings_file.write(" ".join(postings_parts) + "\n")
        
# Function to write normal index to dictionary file
def write_normal_index_to_dict(index_dictionary, dict_file, postings_file):
    # Sort the terms alphabetically
    for term in sorted(index_dictionary.keys()):
        # Get the offset of the postings_file
        offset = postings_file.tell()
        number_of_documents = len(index_dictionary[term])
        dict_file.write(f"{term} {number_of_documents} {offset}\n")
        
        postings_parts = []
        for doc_id, tf in sorted(index_dictionary[term].items()):
            postings_parts.append(f"{doc_id}:{tf}")

        postings_file.write(" ".join(postings_parts) + "\n")       

# Function to Write the Different Document Lengths for Content and Title to the Output Dictionary File
def write_document_lengths(document_lengths, dict_file):
    for doc_id, length in document_lengths.items():
        dict_file.write(f"{doc_id} {length}\n")

def build_index(input_directory, output_file_dictionary, output_file_postings):
    
    index_content_dictionary = {}
    index_title_dictionary = {}
    index_court_dictionary = {}

    # Initialize Porter Stemmer
    porter = PorterStemmer()
    
    # Iterate through all cases in the input directory using the iter_cases generator
    for _, case in enumerate(iter_cases(input_directory)):
        index_content_title_text(case['content'], case['document_id'], index_content_dictionary, porter)
        index_content_title_text(case['title'], case['document_id'], index_title_dictionary, porter)
        index_court(case['court'], case['document_id'], index_court_dictionary)
        
    document_content_lengths = document_length_calculation(index_content_dictionary)
    document_title_lengths = document_length_calculation(index_title_dictionary)
    
    # Write to the output dictionary and postings files
    with open(output_file_dictionary, 'w', encoding='utf-8') as dict_file, \
     open(output_file_postings, 'w', encoding='utf-8') as postings_file:
        dict_file.write("DICTIONARY TERMS FOR CONTENT" + "\n")
        write_positional_index_to_dict(index_content_dictionary, dict_file, postings_file)
        dict_file.write("DICTIONARY TERMS FOR TITLE" + "\n")
        write_positional_index_to_dict(index_title_dictionary, dict_file, postings_file)
        dict_file.write("DICTIONARY TERMS FOR COURT" + "\n")
        write_normal_index_to_dict(index_court_dictionary, dict_file, postings_file)
        dict_file.write("DOCUMENT LENGTHS FOR CONTENT" + "\n")
        write_document_lengths(document_content_lengths, dict_file)
        dict_file.write("DOCUMENT LENGTH FOR TITLE" + "\n")
        write_document_lengths(document_title_lengths, dict_file)

input_directory = output_file_dictionary = output_file_postings = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'i:d:p:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-i': # input directory
        input_directory = a
    elif o == '-d': # dictionary file
        output_file_dictionary = a
    elif o == '-p': # postings file
        output_file_postings = a
    else:
        assert False, "unhandled option"

if input_directory == None or output_file_postings == None or output_file_dictionary == None:
    usage()
    sys.exit(2)

build_index(input_directory, output_file_dictionary, output_file_postings)