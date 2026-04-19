import re
import sys
import getopt
import math
from nltk.stem import PorterStemmer
import bisect
from collections import defaultdict
import heapq
    
def usage():
    print("usage: " + sys.argv[0] + " -d dictionary-file -p postings-file -q file-of-queries -o output-file-of-results")

def normalize_query_text(text):
    """
    Normalize a query token/string the same way as index.py:
    - extract alphabetic chunks
    - lowercase
    - stem
    """
    normalized = []
    subwords = re.findall(r"[A-Za-z]+", text)
    stemmer = PorterStemmer()
    
    for subword in subwords:
        subword = subword.lower()
        normalized.append(stemmer.stem(subword))

    return normalized

# Function to do Search Logic based on the query provided
def run_search(dictionary_file, postings_file, file_of_queries, file_of_output):
    # Use parse_dictionary function to parse the dictionary file into separate sections, returns a dictionary
    parse_dictionary_result = parse_dictionary(dictionary_file)
    query = ""
    
    # Opens the file of query and reads the 1st line as each file only has 1 query
    with open(file_of_queries, "r", encoding="utf-8") as query_file:
        query = query_file.readline().strip()
        
    # Use parse_query function to parse the query and reject invalid queries, returns an array of processed query terms
    parsed_query, mode = parse_query(query)
    
    if not parsed_query:
        with open(file_of_output, "w", encoding="utf-8") as output_file:
            output_file.write("")
        return
    
    # query term frequencies to do TF IDF weighting for the query
    query_tf = {}
    
    # Cache list of dictionary terms for query expansion to avoid repeated retrieval from the dictionary
    cached_dictionary_terms = sorted(parse_dictionary_result["content_dict"].keys())
    
    with open(file_of_output, "w", encoding="utf-8") as results_file, open(postings_file, "r", encoding="utf-8") as postings_file:
        # Handle FREE_TEXT queries first
        if mode == "FREE_TEXT":
            # Court-name exact match take precedence for Free Text Matching
            court_results = get_court_posting_list_if_exact_match(query, parse_dictionary_result["court_dict"], postings_file)
            
            # exact match found just write to results_file and return
            if court_results is not None:
                results_file.write(" ".join(str(x[0]) for x in court_results))
                return
            for term in parsed_query:
                # It should be of type tuple and be length 2
                if not isinstance(term, tuple) or len(term) != 2:
                    continue

                if term[0] == "TERM" and isinstance(term[1], str):
                    normalized_terms = normalize_query_text(term[1])
                    for normalized_term in normalized_terms:
                        query_tf[normalized_term] = query_tf.get(normalized_term, 0) + 1

                elif term[0] == "PHRASE" and isinstance(term[1], list):
                    for phrase_word in term[1]:
                        if isinstance(phrase_word, str):
                            normalized_terms = normalize_query_text(phrase_word)
                            for normalized_term in normalized_terms:
                                query_tf[normalized_term] = query_tf.get(normalized_term, 0) + 1

            # Let N be the total number of documents
            N = len(parse_dictionary_result["content_doc_lengths"])
            
            # Calculate query weights based on TF-IDF weighting, LTC scheme
            content_query_weights = compute_query_weights(
                query_tf,
                parse_dictionary_result["content_dict"],
                N
            )

            title_query_weights = compute_query_weights(
                query_tf,
                parse_dictionary_result["title_dict"],
                N
            )
            
            # If none of the normalized query terms are in the title and content dictionary, just write empty line and return
            if not content_query_weights and not title_query_weights:
                results_file.write("")
                return
            
            # do pseudo relevant feedback withtf-idf cosine similarity for ranking
            content_results = pseudo_relevant_feedback_ranking(
                content_query_weights,
                parse_dictionary_result["content_dict"],
                parse_dictionary_result["content_doc_lengths"],
                postings_file
            )

            title_results = calculate_cosine_similarity(
                title_query_weights,
                parse_dictionary_result["title_dict"],
                parse_dictionary_result["title_doc_lengths"],
                postings_file
            )
            results = combine_field_scores(content_results, title_results)
                
            # Each element in result is a tuple of (doc_id, tf-idf score value)
            results_file.write(" ".join(str(x[0]) for x in results))
            return

        # Handle Boolean Queries
        if mode == "BOOLEAN":
            # Initialize array to store intermediate posting lists for each term, before we do AND operation
            intermediate_posting_list_array = []
 
            for term in parsed_query:
                
                # reset query_tf for each term in the Boolean Query, as each Phrase is treated as a separate component in the Boolean Query
                query_tf = {}
                
                if term[0] == "TERM":
                    # If it is a normal term, we will normalize the term with Porter Stemming and Lowercasing
                    normalized_terms = normalize_query_text(term[1])
                    
                    # If no normalized terms just skip and append empty array
                    if not normalized_terms:
                        intermediate_posting_list_array.append([])
                        continue
                    
                    # Iterate through normalized terms in case theres more than one
                    for normalized_term in normalized_terms:
                        # Temporary Array to store the doc_id of original and expanded terms
                        temporary_array = []
                        # Retrieve the posting list for the expanded term and add it to the intermediate posting list array
                        if normalized_term in parse_dictionary_result["content_dict"]:
                            # add prefix expansions for TERM only as Query Expansion
                            expanded_terms = query_expansion_by_prefix(normalized_term, cached_dictionary_terms)
                            for expanded_term in expanded_terms:
                                _, offset = parse_dictionary_result["content_dict"][expanded_term]
                                posting_list = parse_postings_line(postings_file, offset)
                                if expanded_term == normalized_term:
                                    #  Initialize each relevant doc_id for the term with score value of 1.0 for the actual term
                                    temporary_array.append([(doc_id, 1.0) for doc_id, _ in posting_list])
                                elif expanded_term != normalized_term:
                                    # Since this is an expanded term, we give it lighter weight
                                    temporary_array.append([(doc_id, 0.3) for doc_id, _ in posting_list])
                            intermediate_posting_list_array.append(union_posting_lists_for_query_expansion(temporary_array))
                
                elif term[0] == "PHRASE":
                    # Extract List of Phrasal Terms
                    phrasal_query = term[1]
                    
                    # Reconstrcut the Phrasal Query in continuous text
                    phrase_text = " ".join(term[1])
                    
                    # Check if theres exact match for court name
                    court_results = get_court_posting_list_if_exact_match(
                        phrase_text,
                        parse_dictionary_result["court_dict"],
                        postings_file
                    )
                    
                    # If there is exact match, write to results_file and return
                    if court_results is not None:
                        intermediate_posting_list_array.append(court_results)
                        continue
                    
                    # For each term in the phrasal query, we will do normalization and do query term frequency counting for the phrasal query
                    normalized_phrasal_terms = []
                    for t in phrasal_query:
                        normalized_phrasal_terms.extend(normalize_query_text(t))
                    for normalized_phrasal_term in normalized_phrasal_terms:
                        query_tf[normalized_phrasal_term] = query_tf.get(normalized_phrasal_term, 0) + 1
                        
                    # Let N be the total number of documents 
                    N = len(parse_dictionary_result["content_doc_lengths"])
                    
                    # Calculate query weights based on TF-IDF weighting, LTC scheme
                    content_phrasal_weights = compute_query_weights(
                        query_tf,
                        parse_dictionary_result["content_dict"],
                        N
                    )

                    title_phrasal_weights = compute_query_weights(
                        query_tf,
                        parse_dictionary_result["title_dict"],
                        N
                    )
        
                    content_phrase_results = calculate_cosine_similarity(
                        content_phrasal_weights,
                        parse_dictionary_result["content_dict"],
                        parse_dictionary_result["content_doc_lengths"],
                        postings_file
                    )

                    title_phrase_results = calculate_cosine_similarity(
                        title_phrasal_weights,
                        parse_dictionary_result["title_dict"],
                        parse_dictionary_result["title_doc_lengths"],
                        postings_file
                    )

                    phrasal_query_results = combine_field_scores(
                        content_phrase_results,
                        title_phrase_results,
                        title_weight= 0.5
                    )
                    
                    # Sort by doc_id so that we can do set AND operations using 2 pointers
                    phrasal_query_results = sorted(phrasal_query_results, key=lambda x: x[0])
                    intermediate_posting_list_array.append(phrasal_query_results)
                    
            # Initialize final results array
            final_results = intermediate_posting_list_array[0] if len(intermediate_posting_list_array) > 0 else []
            
            # If the intermediate_posting_list_array is 1, write final results to results file and return to end program
            if len(intermediate_posting_list_array) == 1:
                # Sort by descending score before writing to results_file
                final_results = sorted(final_results, key=lambda x: x[1], reverse=True)
                results_file.write(" ".join(str(x[0]) for x in final_results))
                return
            
            # Repeatedly AND-merge with the remaining posting lists
            for idx in range(1, len(intermediate_posting_list_array)):
                current_list = intermediate_posting_list_array[idx]

                i, j = 0, 0
                merged_results = []

                while i < len(final_results) and j < len(current_list):
                    final_doc_id, final_score = final_results[i]
                    current_doc_id, current_score = current_list[j]

                    if final_doc_id == current_doc_id:
                        merged_results.append((final_doc_id, final_score * current_score))
                        i += 1
                        j += 1
                    elif final_doc_id < current_doc_id:
                        i += 1
                    else:
                        j += 1
                final_results = merged_results

            # Rank final intersection by score descending
            final_results = sorted(final_results, key=lambda x: x[1], reverse=True)
            results_file.write(" ".join(str(x[0]) for x in final_results))
            return
                
# Function to do Cosine Similarity, followed by taking the top 10 as relevant
# Adjust the query vector based on relevance feedback
# Do Cosine Similarity again based on the new query feedback
def pseudo_relevant_feedback_ranking(query_weights, content_dictionary, content_document_length, postings_file):
    # First pass: rank documents and collect doc vectors for pseudo-relevance feedback
    initial_results, doc_vectors = calculate_cosine_similarity(
        query_weights, content_dictionary, content_document_length, postings_file,
        return_doc_vectors=True
    )
    
    # Rocchio pseudo-relevance feedback: treat top-15 as relevant
    pseudo_relevant = [doc_vectors[doc_id] for doc_id, _ in initial_results[:15] if doc_id in doc_vectors]
    if pseudo_relevant:
        # do relevance feedback using rocchio
        expanded_weights = relevance_feedback_by_rocchio(query_weights, pseudo_relevant, [])
        # recalculate similarity scoring using tf-idf scoring with adjusted search query
        results = calculate_cosine_similarity(expanded_weights, content_dictionary, content_document_length, postings_file)
    else:
        results = initial_results
            
    return results

def combine_field_scores(content_scores, title_scores, title_weight=0.5):
    combined = defaultdict(float)

    for doc_id, score in content_scores:
        combined[doc_id] += score

    for doc_id, score in title_scores:
        combined[doc_id] += title_weight * score

    return sorted(combined.items(), key=lambda x: x[1], reverse=True)
        
# Convert raw query term frequencies to ltc TF-IDF weights
# returns a dictionary in the form {term, weights}
def compute_query_weights(query_tf, term_dictionary, N):
    query_weights = {}
    for term, freq in query_tf.items():
        if term not in term_dictionary:
            continue
        df, _ = term_dictionary[term]
        if df == 0:
            continue
        idf = math.log10(N / df)
        query_weights[term] = (1 + math.log10(freq)) * idf
    return query_weights

def calculate_cosine_similarity(query_weights, term_dictionary, doc_length, postings_file, return_doc_vectors = False):
    # Compute the query vector length for normalization using the pre-computed ltc weights
    query_length_squared = sum(w ** 2 for w in query_weights.values())
    query_length = math.sqrt(query_length_squared) if query_length_squared > 0 else 0.0

    # If no query terms have weight, return empty results immediately
    if not query_weights or query_length == 0:
        return ([], {}) if return_doc_vectors else []
    
    scores = {}
     # Initialise doc_vectors only if caller needs them for Rocchio feedback
    doc_vectors = defaultdict(dict) if return_doc_vectors else None

    # For each query term, retrieve its postings list and accumulate dot-product contributions into scores
    for term, w_tq in query_weights.items():
        # If query term is not in ters dictionary, continue
        if term not in term_dictionary:
            continue
        # retrieve offset from term dictionary
        _, offset = term_dictionary[term]
        # retrieve the postings list and parse
        postings_list = parse_postings_line(postings_file, offset)
        # Compute lnc document weight (log tf, no idf) and add to the running score for each document
        for doc_id, tf_d in postings_list:
            w_td = 1 + math.log10(tf_d)
            scores[doc_id] = scores.get(doc_id, 0.0) + w_tq * w_td
            if return_doc_vectors:
                # Store unnormalized lnc weight to be divided by doc length below
                doc_vectors[doc_id][term] = w_td

    # Normalize each document score by its precomputed document length and the query length
    for doc_id in list(scores.keys()):
        length = doc_length.get(doc_id, 0)
        if length != 0 and query_length != 0:
            scores[doc_id] /= (length * query_length)
            if return_doc_vectors:
                # Normalize each term weight in the doc vector by the document length
                for term in doc_vectors[doc_id]:
                    doc_vectors[doc_id][term] /= length
        else:
            scores[doc_id] = 0.0

    # Sort documents by descending score and filter out zero-score documents
    ranked = [(doc_id, score_value) for doc_id, score_value in
              sorted(scores.items(), key=lambda x: x[1], reverse=True)
              if scores[doc_id] > 0]

    if return_doc_vectors:
        return ranked, dict(doc_vectors)
    return ranked
        
# Function to parse dictionary file into separate sections
def parse_dictionary(dictionary_file):
    content_dict = {}
    title_dict = {}
    court_dict = {}
    content_doc_lengths = {}
    title_doc_lengths = {}

    current_section = None

    with open(dictionary_file, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()

            if not line:
                continue

            # Logic to Identify the current section of the dictionary file and parse accordingly
            if line == "DICTIONARY TERMS FOR CONTENT":
                current_section = "content_terms"
                continue
            elif line == "DICTIONARY TERMS FOR TITLE":
                current_section = "title_terms"
                continue
            elif line == "DICTIONARY TERMS FOR COURT":
                current_section = "court_terms"
                continue
            elif line == "DOCUMENT LENGTHS FOR CONTENT":
                current_section = "content_lengths"
                continue
            elif line == "DOCUMENT LENGTH FOR TITLE":
                current_section = "title_lengths"
                continue

            # Split the line into parts and parse based on the current section
            parts = line.split()

            # If current section is content_terms, the line is expected to be in the format: term df offset
            if current_section == "content_terms":
                # format: term df offset
                term = parts[0]
                df = int(parts[1])
                offset = int(parts[2])
                content_dict[term] = (df, offset)

            # If current section is title_terms, the line is expected to be in the format: term df offset
            elif current_section == "title_terms":
                term = parts[0]
                df = int(parts[1])
                offset = int(parts[2])
                title_dict[term] = (df, offset)

            # If current section is court_terms, the line is expected to be in the format: term df offset, but term may contain spaces, so we need to split from the back
            elif current_section == "court_terms":
                # court names may contain spaces, so split from the back
                offset = int(parts[-1])
                df = int(parts[-2])
                term = " ".join(parts[:-2])
                court_dict[term] = (df, offset)

            # If current section is content_lengths, the line is expected to be in the format: doc_id length
            elif current_section == "content_lengths":
                doc_id = parts[0]
                length = float(parts[1])
                content_doc_lengths[doc_id] = length

            # If current section is title_lengths, the line is expected to be in the format: doc_id length
            elif current_section == "title_lengths":
                doc_id = parts[0]
                length = float(parts[1])
                title_doc_lengths[doc_id] = length

    return {
        "content_dict": content_dict,
        "title_dict": title_dict,
        "court_dict": court_dict,
        "content_doc_lengths": content_doc_lengths,
        "title_doc_lengths": title_doc_lengths
    }
    
# Function to parse the query (Sanity Checks, Phrasal Queries and (AND) Queries)
def parse_query(query):
    # Initially split the query into an array of terms based on spaces
    initial_processing_array = query.split()
    processing_array = []
    for initial_term in initial_processing_array:
        if initial_term.startswith('"') and initial_term.endswith('"'):
            split_term = ['"', initial_term[1:-1], '"']
            processing_array += split_term
        elif initial_term.startswith('"'):
            split_term = ['"', initial_term[1:]]
            processing_array += split_term
        elif initial_term.endswith('"'):
            split_term = [initial_term[:-1], '"']
            processing_array += split_term
        else:
            processing_array.append(initial_term)
    # Initialize final processed array for processed queries
    processed_array = []
    # Initialize mode is BOOLEAN, but it can switch to FREE_TEXT if have successive terms with no double quotes
    mode = "BOOLEAN"
    # flag to track the state of the double quotes
    in_quotes = False
    
    # Variable to store the pharsal query when we encounter double quotes
    phrasal_query = []
    
    # Subsequently, we will process the query to identify phrasal queries and AND queries, and store them
    for i in range(len(processing_array)):
        query_term = processing_array[i]
        # If the query term is an AND query, and it is the first term, we will treat it as an error and return an empty array
        if query_term == "AND" and len(processed_array) == 0:
            print("Error: AND operator cannot be the first term in the query")
            return [], None
        
        # If the query term is an AND query and it is the last term, we will treat it as an error and return an empty array
        if query_term == "AND" and i == len(processing_array) - 1:
            print("Error: AND operator cannot be the last term in the query")
            return [], None
        
        if query_term == '"' and in_quotes:
            in_quotes = False
            # In the case the phrasal query ended, we just flush and store the phrasal query in the processed aray
            if phrasal_query:
                processed_array.append(('PHRASE', phrasal_query))
            phrasal_query = []
            continue
        elif query_term == '"' and not in_quotes:
            in_quotes = True
            continue
        
        # Free Text Query if not inquotes and multiple terms appear successively without AND operator
        if query_term != "AND" and len(processed_array) > 0 and processed_array[-1][0] == "TERM" and not in_quotes:
            mode = "FREE_TEXT"
            processed_array.append(('TERM', query_term))
            continue
        
        # If the query is a Free Text Query and the current term is an AND operator, return empty array as it is an invalid query
        if query_term == "AND" and mode == "FREE_TEXT":
            print("Error: AND operator cannot appear in a free text query")
            return [], None
        
        # If the query term is not an AND operator and it is immediately after a phrasal query
        # (indicated by a closing double quote) and the AND operator did not appear before in the query
        if query_term != "AND" and processed_array and processed_array[-1][0] == "PHRASE" and not in_quotes and processing_array[i-1] != "AND":
            print("Error: Missing AND operator between phrasal query and other terms")
            return [], None
    
        else:
            # If term is not an AND operator and we are not in quotes, it is just a normal term
            if query_term != "AND" and not in_quotes:
                # We also store the normal term in the processed array
                processed_array.append(('TERM', query_term))
            
            # If the term is AND operator, we continue as AND operator is Commutative
            elif query_term == "AND":
                continue
            
            # If we are in quotes, keep building the phrasal query
            elif in_quotes:
                phrasal_query.append(query_term)
            
        # If there are unmatched double quotes, we will treat it as an error and return an empty array
        if in_quotes and i == len(processing_array) - 1:
            print("Error: Unmatched double quote in the query")
            return [], None
        
    # If the phrasal query is the only component in the Boolean query, it is an error, return an empty array
    if len(processed_array) == 1 and processed_array[0][0] == "PHRASE":
        print("Error: Phrasal query cannot be the only component in Boolean query")
        return [], None
        
    return processed_array, mode

# Function to read posting file based on offset and parse the line to get the document_id and term frequencies
def parse_postings_line(postings_file, offset):
    # Parses strings like: "(1, 2) (5, 7) (10, 1)"
    line = read_postings_at_offset(postings_file, offset)
    matches = re.findall(r"\(['\"]([^'\"]+)['\"],\s*(\d+)\)", line)
    postings = []

    for doc_id, tf in matches:
        postings.append((str(doc_id), int(tf)))
    return postings

# Function to read posting file based on offset and parse the line
def read_postings_at_offset(postings_file, offset):
    postings_file.seek(offset)
    line = postings_file.readline().strip()
    return line

def query_expansion_by_prefix(query_term, dictionary_terms, max_expansions=5):
    if not query_term or len(query_term) < 4 or not query_term.isalpha():
        return [query_term]

    index = bisect.bisect_left(dictionary_terms, query_term)
    expanded_terms = [query_term]

    while index < len(dictionary_terms):
        candidate = dictionary_terms[index]
        if not candidate.startswith(query_term):
            break
        if candidate != query_term:
            expanded_terms.append(candidate)
            if len(expanded_terms) >= max_expansions + 1:
                break
        index += 1

    return expanded_terms

def relevance_feedback_by_rocchio(query_term, relevant_docs, irrelevent_docs):
    alpha=1.0
    beta=0.65
    gamma=0.15
    k=20

    if irrelevent_docs is None:
        irrelevent_docs = []

    new_query = defaultdict(float)

    # Turn original query into weights
    for term, weight in query_term.items():
        new_query[term] += alpha * weight
    
    # Add weights from relevant docs
    if relevant_docs:
        for doc in relevant_docs:
            for term, weight in doc.items():
                new_query[term] += beta * (weight/len(relevant_docs))

    # Reduced weights from irrelevant docs
    if irrelevent_docs:
        for doc in irrelevent_docs:
            for term,weight in doc.items():
                new_query[term] -= gamma * (weight/len(irrelevent_docs))

    # Remove negative weights
    cleaned_query = {term: max(0.0, weight) for term, weight in new_query.items()}

    # use heapq to get k largest weights and their respective terms
    top_terms = heapq.nlargest(k,cleaned_query.items(),key=lambda x:x[1])
    
    # return it as a dictionary
    return dict(top_terms)

def get_court_posting_list_if_exact_match(query_text, court_dict, postings_file):
    # lower case the query text
    lowered_query = query_text.strip().lower()

    # Check if there in exact match in court_dictionary
    if lowered_query in court_dict.keys():
        _, offset = court_dict[lowered_query]
        posting_list = parse_postings_line(postings_file, offset)
        return [(doc_id, 1.0) for doc_id, _ in posting_list]
    return None
    
# Function to do Union on Query Term with its Expanded Terms
def union_posting_lists_for_query_expansion(intermediate_posting_list):
    merge_postings = {}

    for posting_list in intermediate_posting_list:
        for doc_id, weight in posting_list:
            merge_postings[doc_id] = max(merge_postings.get(doc_id, 0), weight)

    # Ensure sorted by doc_id for AND intersection operation later if needed
    return sorted(merge_postings.items(), key=lambda x: x[0])

dictionary_file = postings_file = file_of_queries = file_of_output = None

try:
    opts, args = getopt.getopt(sys.argv[1:], 'd:p:q:o:')
except getopt.GetoptError:
    usage()
    sys.exit(2)

for o, a in opts:
    if o == '-d':
        dictionary_file = a
    elif o == '-p':
        postings_file = a
    elif o == '-q':
        file_of_queries = a
    elif o == '-o':
        file_of_output = a
    else:
        assert False, "unhandled option"

if dictionary_file == None or postings_file == None or file_of_queries == None or file_of_output == None:
    usage()
    sys.exit(2)

run_search(dictionary_file, postings_file, file_of_queries, file_of_output)