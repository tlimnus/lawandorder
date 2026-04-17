import sys
import getopt
import nltk
#Use Python CSV Module to read the dataset
import csv
import math

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
            
# Function to Normalize Tokens by Lowercasing and Stemming using Porter Stemmer, used in both content and title indexing
def normalize_tokens(tokens, porter):
    return [porter.stem(token.lower()) for token in tokens]

# Function to Add Term to the Index Dictionary, used for both content and title indexing
def add_term(postings_dict, term, doc_id):
    if term not in postings_dict:
        postings_dict[term] = {doc_id: 1}
    elif doc_id not in postings_dict[term]:
        postings_dict[term][doc_id] = 1
    else:
        postings_dict[term][doc_id] += 1
        
# Function to build index for Content of the Document
def index_content(text, doc_id, postings_dict, porter):
    # Use NLTK Tokenizer to Split Document into Sentences
    sentences = nltk.sent_tokenize(text)
    for sentence in sentences:
        words = nltk.word_tokenize(sentence)
        #  Normalize the Tokens by Lowercasing and Stemming using Porter Stemmer
        normalized_words = normalize_tokens(words, porter)
        # Add Each Normalized Word to the Index Dictionary with the Document ID and Term Frequency
        for word in normalized_words:
            add_term(postings_dict, word, doc_id)

# Function to build index for Title of the Document
def index_title(title, doc_id, postings_dict, porter):
    # Use NLTK Tokenizer to Split Title into Words
    title_words = nltk.word_tokenize(title)
    # Normalize the Title Words by Lowercasing and Stemming using Porter Stemmer
    normalized_title_words = normalize_tokens(title_words, porter)
    # Add Each Normalized Title Word to the Index Dictionary with the Document ID and Term Frequency
    for word in normalized_title_words:
        add_term(postings_dict, word, doc_id)
        
# Function to build index for Content of the Document
def index_court(court_name, doc_id, postings_dict, valid_courts):
    if court_name in valid_courts:
        add_term(postings_dict, court_name, doc_id)

# Function to do Document_Length_Calculation for LTC Weighting Scheme
# Logarithm for Term Frequency (TF)
# Logarithm for Inverse Document Frequency (IDF)
# Square Root for Document Length Normalization

def document_length_calculation(index_dictionary, total_documents):
    document_lengths = {}
    for term in index_dictionary:
        for doc_id, tf in index_dictionary[term].items():
            # Calculate the Term Frequency (TF) term for LNC weighting scheme
            tf_weight = 1 + math.log10(tf)
            
            if doc_id not in document_lengths:
                document_lengths[doc_id] = 0
            
            document_lengths[doc_id] += tf_weight ** 2
            
    for doc_id in document_lengths:
        document_lengths[doc_id] = math.sqrt(document_lengths[doc_id])
        
    return document_lengths

# Function to Write the Different Index Dictionaries to the Output Dictionary and Postings Files in the Required Format
def write_index_to_dict(index_dictionary, dict_file, postings_file):
    for term in sorted(index_dictionary.keys()):
        offset = postings_file.tell()
        number_of_documents = len(index_dictionary[term])
        dict_file.write(f"{term} {number_of_documents} {offset}\n")
        document_frequency_pairs = list(sorted(index_dictionary[term].items()))
        postings_file.write(" ".join(str(frequency_pair) for frequency_pair in document_frequency_pairs) + "\n")

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
    
    # List of Courts as Provided in the Dataset
    courts = ['SG Court of Appeal', 'SG Privy Council', 'UK House of Lords', 'UK Supreme Court', 'High Court of Australia', 'CA Supreme Court', 
              'SG High Court', "Singapore International Commercial Court", "HK High Court", "HK Court of First Instance", "UK Crown Court",
              "UK Court of Appeal", "UK High Court","Federal Court of Australia", "NSW Court of Appeal", "NSW Court of Criminal Appeal", "NSW Supreme Court"]
    
    # Total Number of Documents in the Dataset for IDF Calculation
    total_documents = 0
    
    # Iterate through all cases in the input directory using the iter_cases generator
    for _, case in enumerate(iter_cases(input_directory)):
        total_documents += 1
        index_content(case['content'], case['document_id'], index_content_dictionary, porter)
        index_title(case['title'], case['document_id'], index_title_dictionary, porter)
        index_court(case['court'], case['document_id'], index_court_dictionary, courts)
        
    document_content_lengths = document_length_calculation(index_content_dictionary, total_documents)
    document_title_lengths = document_length_calculation(index_title_dictionary, total_documents)
    
    # Write to the output dictionary and postings files
    with open(output_file_dictionary, 'w', encoding='utf-8') as dict_file, \
     open(output_file_postings, 'w', encoding='utf-8') as postings_file:
        dict_file.write("DICTIONARY TERMS FOR CONTENT" + "\n")
        write_index_to_dict(index_content_dictionary, dict_file, postings_file)
        dict_file.write("DICTIONARY TERMS FOR TITLE" + "\n")
        write_index_to_dict(index_title_dictionary, dict_file, postings_file)
        dict_file.write("DICTIONARY TERMS FOR COURT" + "\n")
        write_index_to_dict(index_court_dictionary, dict_file, postings_file)
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