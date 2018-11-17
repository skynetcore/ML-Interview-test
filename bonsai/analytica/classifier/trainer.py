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

import re
from analytica.classifier.bucket import WordBucket

class WordTrainer(object):
    """ Used both for learning (training) documents and for testing documents. 
        The optional parameter learn_from_files has to be set to True, if a classificator 
        should be trained. If it is a test document learn_from_files has to be set to 
        False. """
 
    def __init__(self, vocabulary):
        self.__name = ""
        self.__document_class = None
        self._words_and_freq = WordBucket()
        WordTrainer._vocabulary = vocabulary
    
    def read_document(self,filename, learn=False):
        """ A document is read. It is assumed, that the document is either 
            encoded in utf-8 or in iso-8859... (latin-1).
            The words of the document are stored in a Bag of Words, i.e.         
            self._words_and_freq = WordBucket() """
        try:
            text = open(filename,"r", encoding='utf-8').read()
        except UnicodeDecodeError:
            text = open(filename,"r", encoding='latin-1').read()
        text = text.lower()
        words = re.split(r"\W", text)

        self._number_of_words = 0
        for word in words:
            self._words_and_freq.add_word(word)
            if learn:
                WordTrainer._vocabulary.add_word(word)
                
    def read_data(self,data, learn=False):
        """ A document is read. It is assumed, that the document is either 
            encoded in utf-8 or in iso-8859... (latin-1).
            The words of the document are stored in a Bag of Words, i.e.         
            self._words_and_freq = WordBucket() """
        text = str(data)
        words = re.split(r"\W", text)

        self._number_of_words = 0
        for word in words:
            self._words_and_freq.add_word(word)
            if learn:
                WordTrainer._vocabulary.add_word(word)


    def __add__(self, other):
        """ Overloading the "+" operator. Adding two documents consists 
            in adding the WordBucket of the Documents """
        res = WordTrainer(WordTrainer._vocabulary)
        res._words_and_freq = self._words_and_freq + other._words_and_freq    
        return res
    
    def vocabulary_length(self):
        """ Returning the length of the vocabulary """
        return len(WordTrainer._vocabulary)
                
    def WordsAndFreq(self):
        """ Returning the dictionary, containing the words (keys) with 
        their frequency (values) as contained in the WordBucket attribute 
        of the document"""
        return self._words_and_freq.BagOfWords()
        
    def Words(self):
        """ Returning the words of the Document object """
        d =  self._words_and_freq.BagOfWords()
        return d.keys()
    
    def WordFreq(self,word):
        """ Returning the number of times the word "word" appeared in the 
        document """
        bow =  self._words_and_freq.BagOfWords()
        if word in bow:
            return bow[word]
        else:
            return 0
                
    def __and__(self, other):
        """ Intersection of two documents. A list of words occuring in 
        both documents is returned """
        intersection = []
        words1 = self.Words()
        for word in other.Words():
            if word in words1:
                intersection += [word]
        return intersection

    
class WordPredictor(WordTrainer):
    def __init__(self, vocabulary):
        WordTrainer.__init__(self, vocabulary)
        self._number_of_docs = 0

    def predict(self,word):
        """ returns the probabilty of the word "word" given the class "self" """
        voc_len = WordTrainer._vocabulary.len()
        SumN = 0
        for i in range(voc_len):
            SumN = WordTrainer._vocabulary.WordFreq(word)
            print(str(i))
        N = self._words_and_freq.WordFreq(word)
        erg = 1 + N
        erg /= voc_len + SumN
        return erg

    def __add__(self,other):
        """ Overloading the "+" operator. Adding two predictor objects 
        consists in adding the WordBucket of the predictor objectss """
        res = WordPredictor(self._vocabulary)
        res._words_and_freq = self._words_and_freq + other._words_and_freq 
 
        return res

    def SetNumberOfDocs(self, number):
        self._number_of_docs = number
    
    def NumberOfDocuments(self):
        return self._number_of_docs
