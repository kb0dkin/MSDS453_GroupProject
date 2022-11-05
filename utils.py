#! /bin/env/python
#
# utils.py
# 
# A series of different utility functions for our project, that will likely
# be used for several different purposes. This is things like parsing
# the document corpus, saving the corpus dataframe in python, and more.
#
# To call this, just import utils from the base folder.

import re,string

from nltk.corpus import stopwords
import pandas as pd
import pickle
from os import path
import glob
import numpy as np 

from sklearn.feature_extraction.text import TfidfTransformer

from tkinter import Tk, filedialog
import tkinter as tk



# Separate words, clean out escape keys and punctuation, more
def clean_doc(doc): 
    #split document into individual words
    tokens=doc.split()
    re_punc = re.compile('[%s]' % re.escape(string.punctuation))
    # remove punctuation from each word
    tokens = [re_punc.sub('', w) for w in tokens]
    # remove remaining tokens that are not alphabetic
    tokens = [word for word in tokens if word.isalpha()]
    # filter out short tokens
    tokens = [word for word in tokens if len(word) > 4]
    #lowercase all words
    tokens = [word.lower() for word in tokens]
    # filter out stop words
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]         
    # word stemming    
    # ps=PorterStemmer()
    # tokens=[ps.stem(word) for word in tokens]
    return tokens


# parses a directory of text files, creates a dataframe, then pot
def parse_corpus_dir(dir_name:str = '.', return_df:bool = True, load_pickle:bool = False, save_pickle:bool = False):
    # navigate to the directory we will be scanning
    # should be able to do this without using the root...
    rt = Tk()
    dir_corp = filedialog.askdirectory(master=rt, initialdir=dir_name, title='Select Corpus Directory')
    rt.destroy()

    filenames = glob.glob(f"{dir_corp}{path.sep}*.txt")
    corp_pkl = path.join(dir_corp,"FileDF.pkl")

    if path.exists(corp_pkl) and load_pickle:
        with open(corp_pkl,'rb') as fid:
            file_df = pickle.load(fid)
    else:
        file_df = pd.DataFrame(columns=['Title','Cleaned_List','Full_Text', 'Publication'])


    for filename in filenames:
        with open(filename, 'r', encoding='utf-8') as fid:
            text_block = fid.read()
    
        title = path.splitext(path.split(filename)[-1])[0] # get rid of path and extension info
        if title not in file_df['Title']:
            title_split = title.split('_')
            word_list = clean_doc(text_block) # pull out the individual words, minus stop words etc
            word_list = ' '.join(word_list)
            file_dict = {'Title':title, 'Cleaned_List':[word_list], 'Full_Text':[text_block], 'Publication':title_split[-1]}
            file_df = pd.concat([file_df, pd.DataFrame(file_dict)])

    # reset the indices
    # file_df.reset_index(inplace=True, drop=True, )
    file_df.set_index('Title',inplace=True)

    if save_pickle:
        with open(corp_pkl, 'wb') as fid:
            pickle.dump(file_df)

    if return_df:
        return file_df
