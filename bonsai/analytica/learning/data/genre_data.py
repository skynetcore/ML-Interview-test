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

import pandas as pd
from os import path
from analytica import constants

genre_data_files = {}

def update_genre_filepath(file_name, file_path):
    if path.isfile(file_path):
        file_path = path.abspath(file_path)
          
    genre_data_files[file_name] = file_path


def extract_genre(dir_path, file_name):
        
    raw_file_path =  dir_path + path.sep + file_name
    ''' read raw file with headers '''
    books_data =  pd.read_csv(raw_file_path, usecols=[3,4,6], encoding = "ISO-8859-1", names=["title", "author", "genre"])
    
    ''' write to training file '''
    file_path = dir_path + path.sep + constants.training_file
    books_data.to_csv(file_path)       
    update_genre_filepath(constants.training_file, file_path)
    
    ''' get all genres '''
    genre_list = list(books_data.genre.unique())
    '''generate genre ids '''
    genre_ids = list(range(constants.genre_id_index, constants.genre_id_index + len(genre_list)))
    ''' make frame '''
    genre_frame = pd.DataFrame({"genre_id":genre_ids, "genre":genre_list})
    
    ''' write genre to file '''
    file_path = dir_path + path.sep + constants.genre_file
    genre_frame.to_csv(file_path)
    update_genre_filepath(constants.genre_file, file_path)

def get_training_file_path(file_name):    
    abs_file_path = genre_data_files[file_name]        
    return abs_file_path

''' test '''
if __name__ == '__main__':
    
    pass