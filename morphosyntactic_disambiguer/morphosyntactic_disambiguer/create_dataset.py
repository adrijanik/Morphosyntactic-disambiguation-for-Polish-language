### ALL IMPORTS ###
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
from itertools import islice, chain
import re
import os
from collections import Counter
import itertools
import time
import progressbar
from lxml import objectify, etree
from data_processor import DataProcessor
from disambiguer import Disambiguer
from model import create_model

def take(n, iterable):
    print("[take] Return first n items of the iterable as a list")
    return list(islice(iterable, n))


def create_dataset(location, TRAIN, TRAIN_TAGGED):
    print("[create_dataset] As a result function creates file: input-output-dataset.csv with data in following format: 100 columns with embeddings + one-hot encoding of interpretation + disambiguation (1/0)")
    print("ETAP I - read in data")
    tagged_data_processor = DataProcessor(TRAIN_TAGGED)
    tagged_data_processor.create_words_dictionary()

    print('Data size %d' % len(tagged_data_processor.words))
    print("Unique words: {0}".format(len(Counter(tagged_data_processor.words))))

    words_sample = take(5, tagged_data_processor.words)
    print("\n".join(["{0} -> {1}".format(x, tagged_data_processor.words[x][0]) for x in words_sample]))
    
    data_processor = DataProcessor(TRAIN)
    data_processor.create_words_dictionary(gold=False)
    print('Data size %d' % len(data_processor.words))
    reference_words =(data_processor.words.keys())
    
    print("ETAP II - read in predefined embeddings")
    chunks = pd.read_csv(location + 'pl-embeddings-skip_pure_words.txt', chunksize=1000000, delimiter=' ', header=None, encoding='utf-8')
    embeddings_df = pd.DataFrame()
    embeddings_df = pd.concat(chunk for chunk in chunks).sort_values(0)
    print(embeddings_df.head())
    del embeddings_df[101]
    subset_of_embeddings = embeddings_df.loc[embeddings_df[0].isin(data_processor.words.keys())]
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(np.shape(subset_of_embeddings))
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    print(subset_of_embeddings.head())

    subset_of_embeddings['interpretation'] =  [tagged_data_processor.words[word][0] for word in subset_of_embeddings[0]]
    subset_of_embeddings['disamb'] = [False for i in range(len(subset_of_embeddings))]

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(np.shape(subset_of_embeddings))
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    interpretation = []
    disamb = []
    word_list_with_duplicates = []
    data_tuples = ()
    words_count = Counter()

    i = 0 
    bar = progressbar.ProgressBar(max_value=progressbar.UnknownLength)
    for word_train in list(data_processor.words):
        try:
            for k in data_processor.words[word_train][0]: # iterate over interpretations
                if tagged_data_processor.words[word_train][0].strip() != k[1].strip(): 
                    disamb.append(0)
                else:
                    disamb.append(1)
                word_list_with_duplicates.append(word_train)
                words_count[word_train] += 1
                interpretation.append(k[1])
        except:
            print("Exception!")
            continue
    
        i += 1
        bar.update(i)
    data_tuples = list(zip(word_list_with_duplicates, interpretation, disamb))
    
    subset_of_embeddings['Count'] = subset_of_embeddings[0].map(words_count)

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(np.shape(subset_of_embeddings))
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    del subset_of_embeddings['interpretation']
    del subset_of_embeddings['disamb']
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(np.shape(subset_of_embeddings))
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    subset_of_embeddings.Count = subset_of_embeddings.Count.fillna(0).astype(int)
    count = subset_of_embeddings['Count'].values
    del subset_of_embeddings['Count']

    subset_with_repetitions = pd.DataFrame(np.repeat(subset_of_embeddings.values, count, axis=0))

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Repetitions: {0}".format(np.shape(subset_with_repetitions)))
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")


    sorted_repetitions_df = subset_with_repetitions.sort_values(by=[0])
    data_tuples = sorted(data_tuples)
    test_df = pd.DataFrame([(i[1],i[2]) for i in data_tuples])
    print(test_df.head())
    subset_with_repetitions['interpretation'] = test_df[0]
    subset_with_repetitions['disamb'] = test_df[1]

    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Repetitions: {0}".format(np.shape(subset_with_repetitions)))
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    print("One-hot representation of morphosynthactic forms")
    result = subset_with_repetitions
    result = pd.concat([subset_with_repetitions,pd.get_dummies(subset_with_repetitions['interpretation'])], axis=1)
#    result['one_hot'] = one_hot.fit_transform(subset_with_repetitions['interpretation'])
#    del result[101]
#    del result[102]
#    del result[103]
    del result['interpretation']
    tmp = result['disamb']
    del result['disamb']
    result = pd.concat([result, tmp],axis=1)
    result.head()
    with open('all_columns','w') as f:
        f.write(str(result.columns.tolist()))
        
    result.to_csv('input-output-dataset.csv', encoding='utf-8')


