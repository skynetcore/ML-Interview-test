'''
Created on Nov 15, 2018

@author: derric.lyns
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns
import math
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
        self.books = pd.DataFrame()
        self.users = pd.DataFrame()
        
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
    
    def get_user_shelf_data_from_id(self, book_id):
        unified_data_str = ""
        for i, row_data in self.books.iterrows():
            if str(row_data['bookISBN']) == str(book_id):
                unified_data_str = str(row_data['bookISBN']) + " " #+ str(row_data["bookName"]) #+ " " + str(row_data["author"])
                
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
    
    def get_user_shelf_data_from_list(self, user_id, user_book_list):
        unified_data_str = ""
        pos = 0
        
        for k in range(len(user_book_list)):
            if str(user_id) == str(user_book_list[k][0]):
                pos = k 
                break
        
        books = user_book_list[pos][1]
        
        for i in range(len(books)):
            book_id = books[i]
            for j, row_data in self.books.iterrows():
                if str(row_data['bookISBN']) == str(book_id):
                    book_str = str(row_data['bookISBN']) + " " #+ str(row_data["bookName"]) #+ " " + str(row_data["author"])
                    unified_data_str = unified_data_str + " " + book_str
                    break
        
                
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
    
    def get_user_books(self, user_id, user_book_list):
        pos = 0
        
        for k in range(len(user_book_list)):
            if str(user_id) == str(user_book_list[k][0]):
                pos = k 
                break
        
        books = user_book_list[pos][1]
        
        return books
    
    
    def get_user_shelf_data(self, row_data):
        unified_data_str = str(row_data['bookISBN']) + " " #+ str(row_data["bookName"]) #+ " " + str(row_data["author"])
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
    def write_csv_result(self, file_name, user_id, book_list, user_book_list):
        score = 0.0
        data_str = ""
        
        if not path.exists(file_name):
            with open(file_name, 'a') as result_file:
                result_file.write('[user(s) \ book(s)]')
                for i in range(len(book_list)):
                    data_str = data_str + "," + str(book_list[i])       
                result_file.write(data_str + "\n")
        
        
        print("prediction for user[" + str(user_id) + "] in progress")
        
        likely_books = []
        likely_user_data = self.predict_class_from_data(self.get_user_shelf_data_from_list(user_id, user_book_list))     
        user_books = self.get_user_books(user_id, user_book_list)        
        
        if len(likely_user_data) > 0:
            goodness = 0.0
            
            for i in range(len(likely_user_data)):
                goodness = likely_user_data[i][1]
                chances_of_goodness = 0.000001
                if goodness > 0.001:
                    similar_books = []
                    print("likely user score " + str(goodness))
                    likely_score = goodness
                    similar_user = likely_user_data[i][0]
                    similar_books = self.get_user_books(similar_user, user_book_list)
                    processed_similar_books = []
                    
                    similar_book_found = False                    
                    ''' nullify whats in user books'''
                    for sib in range(len(similar_books)):
                        for uib in range(len(user_books)):
                            if str(similar_books[sib]) == str(user_books[uib]):
                                similar_book_found = True
                                print("similar books removed")
                                break
                        if not similar_book_found:
                            processed_similar_books.append(similar_books[sib])
                            chances_of_goodness = chances_of_goodness + 1
                    
                    goodness = 1/ chances_of_goodness
                    likely_books.append((goodness, processed_similar_books ))
        
        else:
            print("likely user score is 0")
        
        data_str = ""
        
        score = 0.0
        
        with open(file_name, 'a') as result_file:
            result_file.write(str(user_id))
            for i in range(len(book_list)):                
                for bids in range(len(likely_books)):
                                        
                    books = likely_books[bids][1]
                    for k in range(len(books)):
                        if str(books[k]) == str(book_list[i]):
                            score = round(likely_books[bids][0], 2)                            
                            break
                data_str = data_str + "," + str(score)
                score = 0.0                
            result_file.write(data_str + "\n");
        print("> wrote to file")
        
    def generate_results(self, directory_path, levels = 5):
                
        print("results data ")
        results_file = directory_path + path.sep + "results.csv"        
        self.__classifier.load_training_data(directory_path)        
        user_data = pd.read_csv(directory_path + "Users.csv", encoding = "ISO-8859-1")        
        books_data = pd.read_csv(directory_path +"Books.csv", encoding = "ISO-8859-1")
        internet_data = pd.read_csv(directory_path + "UserEvents.csv", encoding = "ISO-8859-1")
        
        ''' chop chop '''
        internet_data = internet_data.dropna(how="all")        
        books_data = books_data.dropna(how="all")
        self.books = books_data
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
        
        actual_user_list = user_data.user
        isbn_list = books_data.bookISBN
        user_books_list = []
        book_list = []
        
        program_start_time = datetime.now()
        '''process all books'''
        for i in range(len(actual_user_list)):
            book_list = []
            for index, user_rows in processed_unified_data.iterrows():
                if str(actual_user_list[i]) == str(user_rows["user"]):
                    book_list.append((user_rows["bookISBN"]))
            user_books_list.append((actual_user_list[i], book_list))                 
        program_time = self.diff(datetime.now(),  program_start_time)            
        print("Completed user book lists in [" + str(program_time) + "] time ")
        
        program_start_time = datetime.now()
        for i in range(len(actual_user_list)):
            write_start_time = datetime.now()
            self.write_csv_result(results_file, actual_user_list[i], isbn_list, user_books_list)
            record_write_time = self.diff(datetime.now(),  write_start_time)                
            print("Completed user["+ str(actual_user_list[i]) +"] record write in [" + str(record_write_time) + "] time ")                        
        program_time = self.diff(datetime.now(),  program_start_time)        
        print("Completed results write in [" + str(program_time) + "] time ")

    
    def generate_training_data(self, directory_path, levels = 5):
        
        ''' read '''
        user_data = pd.read_csv(directory_path + "Users.csv", encoding = "ISO-8859-1")        
        books_data = pd.read_csv(directory_path +"Books.csv", encoding = "ISO-8859-1")
        self.books = books_data
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
        actual_user_list = user_data.user.unique()
        user_list = processed_unified_data.user.unique()
        isbn_list = processed_unified_data.bookISBN.unique()
        
        count = 0
        variance = len(processed_unified_data)
        divisor = 60 # units
        variance = 100 #variance / divisor
            
        prediction_rev_len = 0
        str_shelf_data = ""
        
        
        self.__classifier.load_training_data("D:\\")
        
        print("data i="+str(self.__classifier.get_loop_index()) + " usesr_list length " + str(len(user_list)))
        
        for i in range(self.__classifier.get_loop_index(), len(user_list)):        
            str_shelf_data = ""
            for index, rows in processed_unified_data.iterrows():
                if str(user_list[i]) == str(rows['user']): 
                    str_shelf_data = str_shelf_data + " " + self.get_user_shelf_data(rows)            
                                
            prediction =  self.predict_class_from_data(str_shelf_data)
            #print(str(prediction))                    
            
            if count < variance:                
                #print("prediction "+ str(prediction[0][1]))
                self.__classifier.learn_from_string(str_shelf_data, str(user_list[i]))                
            else:
                if math.ceil(prediction[0][1]) > 0.50:
                    self.__classifier.learn_from_string(str_shelf_data, prediction[0][0])                    
                else:
                    self.__classifier.learn_from_string(str_shelf_data, str(user_list[i]))                    
                
            count = count + 1
            
            if count > variance:
                count = variance + 500
            
            if count > variance:
                if(len(prediction) > prediction_rev_len):
                    #print("^  = " + str(prediction[0][1]), flush=True)
                    print("^", end="", flush=True)
                else:
                    #print("!" + str(prediction[0][1]), flush=True)
                    print("!", end="", flush=True)
            
            prediction_rev_len = len(prediction)
            self.__classifier.set_loop_index(i)
            self.__classifier.save_training_data("D:\\")
            
            #print("shelf data for user [" + str(user_list[i]) + "] " + str_shelf_data)
        #prediction =  self.predict_class_from_data('743202961')
        
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
    genre_obj.generate_training_data("D:\\") 
    genre_obj.generate_results("D:\\")
    program_time = genre_obj.diff(datetime.now(),  program_start_time)
    print("Total time elapsed for processing = " + str(program_time))
    
    pass