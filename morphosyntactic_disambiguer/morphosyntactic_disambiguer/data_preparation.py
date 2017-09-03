import re
import pandas as pd
import numpy as np
from collections import Counter, OrderedDict
from itertools import islice
import itertools
import time
import progressbar
from string import Template

# check encoding if utf-8
import sys
print("Encoding: {0}".format(sys.stdout.encoding))

def read_xcef_data(filename):
    with open(filename) as f:
        content = f.read()
        pattern = '<\?xml version="1\.0" encoding="UTF-8"\?>\s*<\!DOCTYPE cesAna SYSTEM "xcesAnaIPI\.dtd">\s*<cesAna xmlns\:xlink="http\:\/\/www\.w3\.org\/1999\/xlink" version="1\.0" type="lex disamb">\s*<chunkList>\s*(?P<chunks>[\W\s\d\w]+)<\/chunkList>\s*<\/cesAna>'
        chunks_block = re.search(pattern, content)
        if chunks_block:
            all_chunks = chunks_block.groups('chunks')
            pattern = '<chunk type=\"s\">\s*(?P<chunk>[.\w\W\s]+?)<\/chunk>\s*'
            chunks = re.findall(pattern, all_chunks[0])
            return chunks
        return None

def create_dictionary_train(chunks):
    print("Number of chunks: {0}".format(len(chunks)))
    words = {}
    for chunk in chunks:
        pattern = '(?P<token><tok>\s*(?:[\w\W\d.]+?)<\/tok>\s*?)(?:<ns\/>)?'
        tokens = re.findall(pattern, chunk)
        for tok in tokens:
            pattern = '<orth>(?P<orth>.+)<\/orth>\s*(?:[\w\W\d.]+)'
            orth = re.search(pattern, tok)
            x = orth.group('orth')
            pattern = '<lex><base>(?P<base>.+)<\/base><ctag>(?P<ctag>.+)<\/ctag><\/lex>\s*'
            lexes = re.findall(pattern, tok)
            words[x] = [lexes]
    return words
        
    
def create_dictionary_gold(chunks):
    print("Number of chunks: {0}".format(len(chunks)))
    words = {}
    for chunk in chunks:
        pattern = '(?P<token><tok>\s*(?:[\w\W\d.]+?)<\/tok>\s*?)(?:<ns\/>)?'
        tokens = re.findall(pattern, chunk)
        for tok in tokens:
            pattern = '<orth>(?P<orth>.+)<\/orth>\s*(?:[\w\W\d.]+)'
            orth = re.search(pattern, tok)
            x = orth.group('orth')
            pattern = '<lex disamb=\"1\"><base>(?P<base>.+)<\/base><ctag>(?P<ctag>.+)<\/ctag><\/lex>\s*'
            lexes = re.findall(pattern, tok)
            words[x] = [lexes[0][1]]
    return words


def create_dictionary_test(chunks):
    print("Number of chunks: {0}".format(len(chunks)))
    words = {}
    for chunk in chunks:
        pattern = '(?P<token><tok>\s*(?:[\w\W\d.]+?)<\/tok>\s*?)(?:<ns\/>)?'
        tokens = re.findall(pattern, chunk)
        i = 0
        for tok in tokens:
            pattern = '<orth>(?P<orth>.+)<\/orth>\s*(?:[\w\W\d.]+)'
            orth = re.search(pattern, tok)
            x = orth.group('orth')
            pattern = '<lex><base>(?:.+)<\/base><ctag>(?P<ctag>.+)<\/ctag><\/lex>\s*'
            lexes = re.findall(pattern, tok)
            words[x] = ([lexes], i)
            i += 1
    return words

def take(n, iterable):
    "Return first n items of the iterable as a list"
    return list(islice(iterable, n))


def prepare_test_data(file_test,  one_hot):
    
### MUST HAVE - PRESERVE ORDER OF WORDS IN OUTPUT!!! ###

    chunks = read_xcef_data(file_test)
    words = create_dictionary_test(chunks)
    print('Data size %d' % len(words))
    print("Unique words: {0}".format(len(Counter(words))))
    
#    words_sample = take(5, words)
#    print("\n".join(["{0} -> {1}".format(x,words[x][0]) for x in words_sample]))
    
    reference_words = list(words.keys()) # words list in order
    
    ### READ IN EMBEDDINGS ###
    chunks = pd.read_csv('pl-embeddings-skip_pure_words.txt', chunksize=1000000, delimiter=' ', header=None, encoding='utf-8')
    embeddings_df = pd.DataFrame()
    embeddings_df = pd.concat(chunk for chunk in chunks).sort_values(0)
    del embeddings_df[101]
    #embeddings_df.head(30)
    
    ### GET SUBSET OF EMBEDDINGS FOR ANALYZED DATA 
    subset_of_embeddings = embeddings_df.loc[embeddings_df[0].isin(words.keys())]
    print("Subset of embeddings: {0}".format(len(subset_of_embeddings)))
    tmp = subset_of_embeddings
    subset_of_embeddings['interpretation'] =  [words[word][0][0] for word in tmp[0]]
    print("Subset of embeddings head: {0}".format(subset_of_embeddings.head()))
    
    word_list_with_duplicates = []
    interpretation = []
    index = []
    def create_series_for_df():
        i = 0
        bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
        for word_train in reference_words:
            try:
                #print(words_train[word_train][0])
                for k in words[word_train][0][0]: # iterate over interpretations
                    word_list_with_duplicates.append(word_train)
                    interpretation.append(k)
                    index.append(words[word_train][1])
            except:
                continue
                
            i += 1
            bar.update(i)
    
    create_series_for_df()
    
    ### PREPARE TRAINING DATA WITH CORRECT AND NOT CORRECT DISAMBIGUATIONS ###
    words_count = Counter(word_list_with_duplicates)
    subset_of_embeddings['Count'] = subset_of_embeddings[0].map(words_count)
    subset_of_embeddings.Count = subset_of_embeddings.Count.fillna(0).astype(int)
    
    subset_with_repetitions = pd.DataFrame(np.repeat(subset_of_embeddings.values, subset_of_embeddings['Count'].values, axis=0))
    
    data_tuples = list(zip(word_list_with_duplicates, interpretation, index))
    print(data_tuples[:10])
    
    sorted_repetitions_df = subset_with_repetitions.sort([0])
    
    data_tuples = sorted(data_tuples)
    test_df = pd.DataFrame([i[1], i[2]] for i in data_tuples )
    #test_df.head()
    subset_with_repetitions['interpretation'] = test_df[0]
    subset_with_repetitions['index'] = test_df[1]

#    subset_with_repetitions['disamb'] = test_df[1]
    #subset_with_repetitions.tail(10)
    
    print("One-hot representation of morphosynthactic forms")
    result = subset_with_repetitions
    result['one_hot'] = one_hot.fit_transform(subset_with_repetitions['interpretation'])
#    result = pd.concat([subset_with_repetitions,pd.get_dummies(subset_with_repetitions['interpretation'])], axis=1)
    print(result.columns)
    del result[100]
    del result[101]
    del result[102]
    del result[u'index']
    #del result[103]
    del result['interpretation']
    input_file = 'input_test.csv'
    result.to_csv(input_file, encoding='utf-8')
    print("Test data saved successfully in: {0}".format(input_file))

