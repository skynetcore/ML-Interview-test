'''
Created on Nov 15, 2018

@author: derric.lyns
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns
from datetime import datetime
from dateutil.relativedelta import relativedelta
from pandas.core.reshape.pivot import pivot_table
from _datetime import date

from analytica.classifier.words import Words
from analytica.learning.data import genre_data
from analytica.learning import constants
from os import path

import pandas as pd
import nltk
from nltk.corpus import wordnet
import re


''' Global data '''
unified_data = pd.DataFrame()
isbn_list = pd.DataFrame()


class NaiveBucket(object):
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
        self.__largest_shelf = 0       
        
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
    
    def minimize_list(self, list_data, max_num):        
        #loop
        if max_num < len(list_data):
            list_data = list_data[:max_num]    
        
        return list_data
    
    def process_affinity_score(self, data_list):
        #['checkout' 'add to cart' 'view' 'like' 'interact' 'dislike']
        user_impressionn  = ["checkout", 'add to cart', 'view', 'like', 'interact', 'dislike']
        impression_score  = [      4   ,      4       ,   1   ,    5  ,    3      ,    -17    ]
        # comment below for production
        # impression_score  = [      0.5   ,      0.5       ,   0   ,    1  ,    0      ,    -1    ]
        
        ''' process score '''
        unified_data = pd.DataFrame(data_list)
        unified_data['score'] = unified_data.index
        
        for x, rows in unified_data.iterrows():
            #print("rows - impression " + rows["impression"])
            for i in range(0, len(user_impressionn)):
                if rows["impression"] == user_impressionn[i]:
                    unified_data.at[x,"score"] = impression_score[i]
                    break            
            #print('row updated')
            
        #print(unified_data.head(n=10))
        return unified_data
    
    def get_user_shelf_data(self, row_data):
        unified_data_str = str(row_data['bookISBN']) + " " + str(row_data["bookName"]) #+ " " + str(row_data["author"])
        #print(unified_data_str)
        sdata = str(unified_data_str).encode(encoding='utf_8')
        sdata = str(sdata).lower()
        sdata = self.refine_string(sdata)
        #sdata = self.refine_numbers(sdata)
        sdata = self.unique_string(sdata)
        sdata = self.remove_single_chars(sdata)
        #sdata = self.remove_tri_chars(sdata)
        #print(sdata)
        return sdata
    
    ''' first write header then data '''
    def write_csv_result(self, file_name, user_id, book_list):
        score = 0.0
        
        if not path.exists(file_name):
            with open(file_name, 'a') as result_file:
                result_file.write('[user(s) \ book(s)]')
                for i in range(len(book_list)):
                    result_file.write(','+ str(book_list[i]))
                result_file.write('\n')
        
        with open(file_name, 'a') as result_file:
            result_file.write(str(user_id))
            for i in range(len(book_list)):
                data = self.predict_class_from_data(book_list[i])
                for k in range(len(data)):
                    if str(data[k][0]) == str(user_id):
                        score = data[k][1] + score             
                result_file.write(','+ str(score))
            result_file.write("\n");
        print("> wrote to file")

    
    def generate_training_data(self, directory_path, levels = 5):
        
        ''' read '''
        user_data = pd.read_csv(directory_path + "Users.csv", encoding = "ISO-8859-1")
        books_data = pd.read_csv(directory_path +"Books.csv", encoding = "ISO-8859-1")
        internet_data = pd.read_csv(directory_path + "UserEvents.csv", encoding = "ISO-8859-1")
        
        ''' chop chop '''
        internet_data = internet_data.dropna(how="all")
        books_data = books_data.dropna(how="all")
        user_data = user_data.dropna(how="all")
        
        ''' display impression '''
        impression = internet_data.impression.unique()
        # print(impression)
        
        '''join has not outer, how makes outer and inner this is simple no NaN product '''
        user_events = pd.merge(internet_data, books_data, left_on="bookId", right_on="bookISBN")
            
        ''' display'''
        unified_data = pd.merge(user_events, user_data, left_on="user", right_on="user")
        #print(unified_data.head(n=10))
        
        ''' process user data for 30 '''
        processed_unified_data = self.process_affinity_score(unified_data)
        
        ''' isbn_list and user_list '''
        user_list = processed_unified_data.user.unique()
        isbn_list = processed_unified_data.bookISBN.unique()
        
        count = 0
        variance = len(processed_unified_data)
        divisor = 100 # units
        variance = variance / divisor
            
        prediction_rev_len = 0
        str_shelf_data = ""
        for i in range(len(user_list)):
            str_shelf_data = ""
            for index, rows in processed_unified_data.iterrows():
                if user_list[i] == rows['user']: 
                    str_shelf_data = str_shelf_data + " " + self.get_user_shelf_data(rows)            
                                
            prediction =  self.predict_class_from_data(str_shelf_data)
            #print(str(prediction))                    
            
            if count < variance:                
                #print("prediction "+ str(prediction[0][1]))
                self.__classifier.learn_from_string(str_shelf_data, str(user_list[i]))                
            else:
                if prediction[0][1] > 0.50:
                    self.__classifier.learn_from_string(str_shelf_data, prediction[0][0])                    
                else:
                    self.__classifier.learn_from_string(str_shelf_data, str(user_list[i]))                    
                
            count = count + 1
            
            if count > variance:
                count = variance + 500
            
            if count > variance:
                if(len(prediction) > prediction_rev_len):
                    print("^  = " + str(prediction[0][1]), flush=True)
                else:
                    print("!" + str(prediction[0][1]), flush=True)
            
            prediction_rev_len = len(prediction)
            
            #print("shelf data for user [" + str(user_list[i]) + "] " + str_shelf_data)
        #prediction =  self.predict_class_from_data('743202961')
        self.__classifier.save_training_data("D:\\")
        
        program_start_time = datetime.now()
        for i in range(len(user_list)):
            self.write_csv_result("results.csv", user_list[i], isbn_list)        
                
        program_time = self.diff(datetime.now(),  program_start_time)    
        #print("Total time elapsed for processing = " + str(program_time))
        print("Completed")
        
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
    
    
    def predict_class_from_data(self, data, similarity = False):
        sdata = str(data).encode(encoding='utf_8')
        sdata = str(sdata).lower()
        sdata = self.refine_string(sdata)
        #sdata = self.refine_numbers(sdata)
        sdata = self.unique_string(sdata)
        sdata = self.remove_single_chars(sdata)
        sdata = self.remove_tri_chars(sdata)
        
        if similarity:
            sdata = self.prepare_thesaurus(sdata)
            
        #print("requested title " + title +" processed to " + sdata)
        return self.__classifier.predict(sdata)
    
    def diff(self, t_a, t_b):
        t_diff = t_a - t_b  # later/end time comes first!
        return t_diff

'''
   DONE!!         
            
'''
    
if __name__ == '__main__':
    
    genre_obj = NaiveBucket()
    program_start_time = datetime.now()
    genre_obj.generate_training_data("data_old\\")
    program_time = genre_obj.diff(datetime.now(),  program_start_time)
    print("Total time elapsed for processing = " + str(program_time))
    
    pass