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

import os
import pickle
from analytica.classifier.bucket import WordBucket
from analytica.classifier.trainer import WordTrainer
from analytica.classifier.trainer import WordPredictor

class Words(object):
    def __init__(self):
        self.__document_classes = {}
        self.__vocabulary = WordBucket()
    
    def load_training_data(self, file_path):
        '''
        pickle_file = open(file_path + os.path.sep + "document_classes.pkl", 'rb')
        pkl_object = pickle.load(pickle_file)
        self.__document_classes = pkl_object
        
        pickle_file = open(file_path + os.path.sep + "all_classes.pkl", 'rb')
        pkl_object = pickle.load(pickle_file)
        self.__vocabulary = pkl_object
        '''
        pickle_file = open(file_path + os.path.sep + "all_classes.pkl", 'rb')
        pkl_object = pickle.load(pickle_file)
        self.__document_classes = pkl_object.__document_classes
        self.__vocabulary = pkl_object.__vocabulary
    
    def save_training_data(self, file_path):
        '''
        pickle_file = open(file_path + os.path.sep + "document_classes.pkl", 'wb')
        pickle.dump(self.__document_classes, pickle_file, pickle.HIGHEST_PROTOCOL)
        
        pickle_file = open(file_path + os.path.sep + "all_classes.pkl", 'wb')
        pickle.dump(self.__vocabulary, pickle_file, pickle.HIGHEST_PROTOCOL)
        '''
        pickle_file = open(file_path + os.path.sep + "all_classes.pkl", 'wb')
        pickle.dump(self, pickle_file, pickle.HIGHEST_PROTOCOL)
                
    def sum_words_per_category(self, dclass):
        """ The number of times all different words of a dclass appear in a class """
        total = 0
        for word in self.__vocabulary.Words():
            WaF = self.__document_classes[dclass].WordsAndFreq()
            if word in WaF:
                total +=  WaF[word]
        return total
    
    def learn_from_files(self, directory, dclass_name):
        """ directory is a path, where the files of the class with the name 
            dclass_name can be found """
        x = WordPredictor(self.__vocabulary)
        dirlist = os.listdir(directory)
        for file in dirlist:
            d = WordTrainer(self.__vocabulary)
            #print(directory + "/" + file)
            d.read_document(directory + "/" +  file, learn = True)
            x = x + d
        self.__document_classes[dclass_name] = x
        print("length of dir " + str(len(dirlist)))
        x.SetNumberOfDocs(len(dirlist))
    
    def learn_from_string(self, string_data, dclass_name):
        """ directory is a path, where the files of the class with the name 
            dclass_name can be found """
        x = WordPredictor(self.__vocabulary)        
        d = WordTrainer(self.__vocabulary)
        #print(directory + "/" + file)
        d.read_data(string_data, learn = True)
        x = x + d
        self.__document_classes[dclass_name] = x
        #print("length of dir " + str(len(dir)))
        x.SetNumberOfDocs(1)
    
    def predict(self, doc, dclass = ""):
        """Calculates the probability for a class dclass given a document doc"""
        if dclass:
            sum_dclass = self.sum_words_per_category(dclass)
            prob = 0
        
            d = WordTrainer(self.__vocabulary)
            """ check if its a file other wise train from string """
            if os.path.isfile(doc):
                d.read_document(doc)
            else:
                d.read_data(doc)
                
            for j in self.__document_classes:
                sum_j = self.sum_words_per_category(j)
                prod = 1
                for i in d.Words():
                    wf_dclass = 1 + self.__document_classes[dclass].WordFreq(i)
                    wf = 1 + self.__document_classes[j].WordFreq(i)
                    r = wf * sum_dclass / (wf_dclass * sum_j)
                    prod *= r
                prob += prod * self.__document_classes[j].NumberOfDocuments() / self.__document_classes[dclass].NumberOfDocuments()
            if prob != 0:
                return 1 / prob
            else:
                return -1
        else:
            prob_list = []
            for dclass in self.__document_classes:
                prob = self.predict(doc, dclass)
                prob_list.append([dclass,prob])
            prob_list.sort(key = lambda x: x[1], reverse = True)
            return prob_list