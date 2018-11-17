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
import analytica.classifier.train_classifier as tc
# Data Manipulation Libraries
import collections as c

RANDOM_STATE = 42

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
        self.__classifier = tc.ClassifierComposite()
        
    def generate_training_data(self, directory_path):
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
        
        features_train, features_test, labels_train, labels_test = \
        tc.generate_panda_test_train(training_books_data, RANDOM_STATE)
        
        self.__classifier.train_svm_minimum(features_train, features_test,
                                    labels_train, labels_test, RANDOM_STATE)
        
        print("classification completed")
    
    
    def pred_genre(self, title, author):
        """
        Predicts a book's genre using the set classifier.
    
        :param isbn: The ISBN of the book being assessed.
        :param clf_composite: A Classifier Composite object containing references
                              to the various classifiers.
        :return:
        """
        # Extract and format strings
        t_d = c.Counter(title.split())
        a_d = c.Counter([author.replace(' ', '_')])
    
        # Results
        t_c = self.__classifier.nlp_title.classify(t_d)
        a_c = self.__classifier.nlp_author.classify(a_d)
        p_c = self.__classifier.clf.predict([[t_c, a_c]])[0]
    
        print('Title: ' + title)
        print('Title Class: ' + self.__genre_list[t_c])
        print('Author Class: ' + self.__genre_list[a_c])
        print('Predicted Class: ' + self.__genre_list[p_c])
        print('')
    '''
       DONE!!         
                
    '''