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

class WordBucket(object):
    """ Implementing a bag of words, words corresponding with their 
        frequency of usages in a "document" for usage by the Document 
        class, Classified class and the Pool class."""
    
    def __init__(self):
        self.__number_of_words = 0
        self.__bag_of_words = {}
        
    def __add__(self, other):
        """ Overloading of the "+" operator to join two WordBucket """
        erg = WordBucket()
        sumofwords = erg.__bag_of_words
        for key in self.__bag_of_words:
            sumofwords[key] = self.__bag_of_words[key]
            if key in other.__bag_of_words:
                sumofwords[key] += other.__bag_of_words[key]
        for key in other.__bag_of_words:
            if key not in sumofwords:
                sumofwords[key] = other.__bag_of_words[key]
        return erg
        
    def add_word(self, word):
        """ A word is added in the dictionary __bag_of_words"""
        self.__number_of_words += 1
        if word in self.__bag_of_words:
            self.__bag_of_words[word] += 1
        else:
            self.__bag_of_words[word] = 1
    
    def len(self):
        """ Returning the number of different words of an object """
        return len(self.__bag_of_words)
    
    def Words(self):
        """ Returning a list of the words contained in the object """
        return self.__bag_of_words.keys()
    
        
    def BagOfWords(self):
        """ Returning the dictionary, containing the words (keys) with their 
            frequency (values)"""
        return self.__bag_of_words
        
    def WordFreq(self,word):
        """ Returning the frequency of a word """
        if word in self.__bag_of_words:
            return self.__bag_of_words[word]
        else:
            return 0
