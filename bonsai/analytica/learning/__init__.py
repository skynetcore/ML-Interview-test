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

from analytica.learning.data import genre_data
from analytica import constants
from analytica.learning.genre import Genre
from os import path

''' test '''
if __name__ == '__main__':
    
    '''
    genre_data.extract_genre( "data", constants.raw_training_file)
    file = genre_data.get_training_file_path(constants.genre_file)
    print("genre file path "+ file)
    
    file = genre_data.get_training_file_path(constants.training_file)
    print("training file path "+ file)
    '''
    
    genre_obj = Genre()
    genre_obj.generate_training_data("data", levels=500)

    '''    
    genre_obj.pred_genre("Harry Potter", "Rowling")
    '''
    
    genre_obj.predict_genre_from_title("harry potter", True)
    genre_obj.predict_genre_from_title("angels and demons", True)
    genre_obj.predict_genre_from_title("time machine", True)
    
    pass