'''
/*
 * Copyright (C) 2018 Derric Lyns [derriclyns@gmail.com]
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
'''

from analytica.classifier.words import Words
from analytica.learning.data import genre_data
from analytica.learning import constants
from os import path

import pandas as pd
import nltk
from nltk.corpus import wordnet
import re

class Genre(object):
    '''
    classdocs
    '''
    def __init__(self):
        '''
        Constructor
        '''
        self.__genre_list = []
        self.__trained_model = ""
        self.__classifier = Words()
        #nltk.download()
        nltk.download('wordnet')
        
    def unique_string(self, string_data):        
        list_data = str(string_data).split()
        ulist = []
        [ulist.append(x) for x in list_data if x not in ulist]
        refined_string = ' '.join(ulist)
        return refined_string
    
    def remove_single_chars(self, string_data):        
        list_data = str(string_data).split()
        ulist = []
        [ulist.append(x) for x in list_data if len(x)>1]
        refined_string = ' '.join(ulist)
        return refined_string
    
    def remove_tri_chars(self, string_data):        
        list_data = str(string_data).split()
        ulist = []
        [ulist.append(x) for x in list_data if len(x)>3]
        refined_string = ' '.join(ulist)
        return refined_string
    
    def refine_numbers(self, string_data):
        refined_data = re.sub(' \d', ' numbers ', str(string_data))
        return refined_data
    
    def refine_string(self, string_data):
        refined_data = re.sub('[^A-Za-z0-9]+', ' ', str(string_data))
        refined_data = str(refined_data).replace('  ', ' ')
        return refined_data
    
    def generate_string_from_list(self, data_list):
        stringed = ""
        for i in range(len(data_list)):
            sdata = str(data_list[i]).encode(encoding='utf_8')
            sdata = str(sdata).lower()
            sdata = self.refine_string(sdata)
            sdata = self.refine_numbers(sdata)
            sdata = self.unique_string(sdata)
            sdata = self.remove_single_chars(sdata)
            sdata = self.remove_tri_chars(sdata)
            #print("sdata " + str(sdata))            
            stringed = stringed + str(sdata) + " "
        
        ''' again unique '''
        stringed = self.unique_string(stringed)
        #print("Unique "+ stringed)
        return stringed
    
    def append_authors_and_title(self, authors_string, titles_string):
        concat_string = "authors " + authors_string + " titles " + titles_string
        #concat_string = self.unique_string(concat_string)
        return concat_string
    
    def minimize_list(self, list_data, max_num):        
        #loop
        if max_num < len(list_data):
            list_data = list_data[:max_num]    
        
        return list_data
        
    def generate_training_data(self, directory_path, levels = 5):
        '''
        Pre Training extracts genre and genre data set
        '''
        self.__trained_model = directory_path + path.sep + constants.trained_model        
        genre_data.extract_genre( directory_path, constants.raw_training_file)
        books_genre = pd.read_csv(genre_data.get_training_file_path(constants.genre_file), encoding = "ISO-8859-1")
        self.__genre_list = list(books_genre.genre)
        
        '''
        print(self.__genre_list)
        '''        
        books_data = genre_data.get_training_file_path(constants.training_file)
        
        '''
        now extract the genre data for training
        '''
        training_books_data = pd.read_csv( books_data, encoding = "ISO-8859-1")
        
        for i in range(len(self.__genre_list)):
            genre_books_table = training_books_data.loc[ training_books_data["genre"] == self.__genre_list[i]]
            genre_book_titles = genre_books_table["title"]
            genre_book_authors = genre_books_table["author"]
            
            ''' debug 
            print("processing " + self.__genre_list[i])
            print(genre_book_titles.head(1))
            print(genre_book_authors.head(1))
            '''
            
            ''' column to array, then to string '''
            genre_authors_list = list(genre_book_authors)
            genre_authors_list = self.minimize_list(genre_authors_list, levels)
            
            genre_title_list = list(genre_book_titles)
            genre_title_list = self.minimize_list(genre_title_list, levels)
            
            ''' now string '''            
            string_genre_authors = self.generate_string_from_list(genre_authors_list)
            string_genre_title = self.generate_string_from_list(genre_title_list)            
            string_genre_training = self.append_authors_and_title(string_genre_authors, string_genre_title)            
            #print("Train Data [Genre:" + self.__genre_list[i] + "] Data : [" + string_genre_training + "]")
            
            
            ''' train title '''
            self.__classifier.learn_from_string(string_genre_training, self.__genre_list[i])            
            print("learning complete for genre " + self.__genre_list[i])
        
        print("classification completed")
        
    def prepare_word_synonyms(self, word):
        syns = []
        syn_sets = wordnet.synsets(word)
        i = 0
        
        for syn_set in syn_sets:
            syn_data = [n.replace('_', ' ') for n in syn_set.lemma_names()]
            syn_data = self.generate_string_from_list(syn_data)
            syns.append(syn_data)
            i = i + 1
            #if i > 3:
            #    break

        string_syns = self.generate_string_from_list(syns)
        print("Thesaurus " + string_syns)
        return string_syns
        
    def prepare_thesaurus(self, string_data):
        words = str(string_data).split(' ')
        thesaurical = ""
        for i in range(0, len(words)):
            thesaurical = thesaurical + " " + self.prepare_word_synonyms(words[i])
            
        return thesaurical
    
    
    def predict_genre_from_title(self, title, similarity = False):
        sdata = str(title).encode(encoding='utf_8')
        sdata = str(sdata).lower()
        sdata = self.refine_string(sdata)
        sdata = self.refine_numbers(sdata)
        sdata = self.unique_string(sdata)
        sdata = self.remove_single_chars(sdata)
        sdata = self.remove_tri_chars(sdata)
        
        if similarity:
            sdata = self.prepare_thesaurus(sdata)
            
        #print("requested title " + title +" processed to " + sdata)
        print("Prediction "+ str(self.__classifier.predict(sdata)))
    
'''
   DONE!!         
            
'''