'''
Created on Nov 18, 2018

@author: derric.lyns
'''

import pandas as pd
import os

if __name__ == '__main__':
    
    directory_path = "data_old\\"
    
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
    processed_unified_data = unified_data #self.process_affinity_score(unified_data)
    
    ''' isbn_list and user_list '''
    user_list = processed_unified_data.user.unique()
    user_data = user_data.user.unique()
    
    print("user_data length = " + str(len(user_data))+ " user_list length " + str(len(user_list)))
    
    invalid_users = []
    valid_users = []
    found = True
    for i in range(len(user_data)):
        found = False
        
        for k in range(len(user_list)):
            if user_data[i] == user_list[k] :
                found = True                
                break
        
        if found == False:
            invalid_users.append(user_data[i])
        else:
            valid_users.append(user_data[i])
    
    
    print("users not in list : " + str(len(invalid_users)))
    invalid_users_df = pd.DataFrame(invalid_users)
    valid_users_df = pd.DataFrame(valid_users)
    invalid_users_df.to_csv("invalid_users.csv")
    valid_users_df.to_csv("valid_users.csv")
    print("completed ..")
        
    
    pass