import pandas as pd
import matplotlib.pyplot as plt  
import seaborn as sns
from os import path
from datetime import datetime
from distutils.file_util import write_file


''' Global data '''
unified_data = pd.DataFrame()
isbn_list = pd.DataFrame()


def plot_data(data_list, column_name):
    sns.set_style('dark')
    plt.figure(figsize=(8,6))  
    plt.rcParams['patch.force_edgecolor'] = True  
    data_list[column_name].hist(bins=50)
    plt.show()

def xyplot_data(data_list, xcolumn_name, ycolumn_name):
    sns.set_style('dark')
    #plt.figure(figsize=(8,6))  
    plt.rcParams['patch.force_edgecolor'] = True  
    sns.jointplot(x=xcolumn_name, y=ycolumn_name, data=data_list, alpha=0.8)
    plt.show()

def recommend_books_from(user_id, book_id, user_book_rating_table, ratings_mean_count):
    
    book_ratings = user_book_rating_table[book_id]   
    movies_like_book = user_book_rating_table.corrwith(book_ratings)

    corr_book = pd.DataFrame(movies_like_book, columns=['Correlation'])  
    corr_book.dropna(inplace=True)
    
    corr_book = corr_book.sort_values('Correlation', ascending=False)
    corr_book = corr_book.join(ratings_mean_count['score_counts']) 
    corr_book = corr_book[corr_book['score_counts']> 1 ].sort_values('Correlation', ascending=False)
    
    #print("user_id [" + str(user_id) + "]  " + str(corr_book))

    return corr_book

'''
['checkout' 'add to cart' 'view' 'like' 'interact' 'dislike']
'''
def process_affinity_score(data_list):
    #['checkout' 'add to cart' 'view' 'like' 'interact' 'dislike']
    user_impressionn  = ["checkout", 'add to cart', 'view', 'like', 'interact', 'dislike']
    impression_score  = [      5   ,      4       ,   1   ,    6  ,    2      ,    -1    ]
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


def get_book_score(user_id, book_id, recommended_books):
    score = 0

    # user_id, book_id
    for i in range(len(recommended_books)):
        if user_id == recommended_books[i][0]:
            if book_id == recommended_books[i][1]:
                books = recommended_books[i][2]
                for k in range(len(books)):
                    score = score + k        
    return score

''' first write header then data '''
def write_csv_result(file_name, user_id, recommended_books, book_list):
    
    if not path.exists(file_name):
        with open(file_name, 'a') as result_file:
            result_file.write('[user(s) \ book(s)]')
            for i in range(len(book_list)):
                result_file.write(','+ str(book_list[i]))
            result_file.write('\n')
    
    with open(file_name, 'a') as result_file:
        result_file.write(str(user_id))
        for i in range(len(book_list)):
            score = get_book_score(user_id, book_list[i], recommended_books)
            result_file.write(','+ str(score))
        result_file.write("\n");
    print("> wrote to file")
    

def pre_process_data():
    # settings
    pd.options.display.width = None
    
    # read
    user_data = pd.read_csv("data_old\\Users.csv", encoding = "ISO-8859-1")
    books_data = pd.read_csv("data_old\\Books.csv", encoding = "ISO-8859-1")
    internet_data = pd.read_csv("data_old\\UserEvents.csv", encoding = "ISO-8859-1")
    
    # chop chop
    internet_data = internet_data.dropna(how="all")
    books_data = books_data.dropna(how="all")
    user_data = user_data.dropna(how="all")
    
    # join has not outer, how makes outer and inner this is simple no NaN produced
    user_events = pd.merge(internet_data, books_data, left_on="bookId", right_on="bookISBN")        
    # display
    unified_data = pd.merge(user_events, user_data, left_on="user", right_on="user")
    #print(unified_data.head(n=10))
    
    ''' process user data for 30 '''
    # process the data
    processed_unified_data = process_affinity_score(unified_data)
    #print(processed_unified_data)    
    ''' tested ok
    data = processed_unified_data.groupby('bookName')['score'].mean()
    print(data.head())
    
    data = processed_unified_data.groupby('bookName')['score'].mean().sort_values(ascending=False).head() 
    print(data.head())
    
    data = processed_unified_data.groupby('bookName')['score'].count().sort_values(ascending=False).head() 
    print(data.head())
    '''
    # now 
    ratings_score_mean = pd.DataFrame(processed_unified_data.groupby('bookISBN')['score'].mean())
    #print(ratings_score_mean)
    ratings_score_mean['score_counts'] = pd.DataFrame(processed_unified_data.groupby('bookISBN')['score'].count())
    #print(ratings_score_mean)
    
    #plot_data(ratings_mean_count, "rating_counts")
    #plot_daa(ratings_mean_count, "score")
    #xyplot_data(ratings_mean_count, "score", "rating_counts")    
    user_book_rating = processed_unified_data.pivot_table(index='user', columns='bookISBN', values='score') 
    #print(user_book_rating)
    
    isbn_list = processed_unified_data.bookISBN.unique()
    user_list = processed_unified_data.user.unique()
     
    recommended_books = []
    program_start_time = datetime.now()
    for i in range(len(user_list)):
        recommended_books = []
        for k in range(len(isbn_list)):
            likely_books = recommend_books_from(user_list[i], isbn_list[k], user_book_rating, ratings_score_mean)                    
            recommended_books.append((user_list[i], isbn_list[k], likely_books ))
            #print("writing to file for user "+str(recommended_books))    
        write_csv_result("results.csv", user_list[i], recommended_books, isbn_list)
        recommended_books.clear()
            
    program_time = diff(datetime.now(),  program_start_time)    
    #print("Total time elapsed for processing = " + str(program_time))
    print("Completed")

def diff(t_a, t_b):
    t_diff = t_a - t_b  # later/end time comes first!
    return t_diff

pre_process_data()