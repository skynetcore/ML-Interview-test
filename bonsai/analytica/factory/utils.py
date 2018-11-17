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

def generate_book_genre():
    # generate data frame
    df = pd.DataFrame(columns=['genre_id','genre'])
    
    df.loc[0] =  [100, "Arts, Film & Photography"]
    df.loc[1] =  [100, "Action & Adventure"]    
    df.loc[2] =  [102, "Biographies, Diaries & Memoirs"]    
    df.loc[3] =  [103, "Business, Money & Economics"]    
    df.loc[4] =  [104, "Calendars"]    
    df.loc[5] =  [105, "Childrens Books"]
    df.loc[6] =  [106, "Comics & Cartoons"]
    df.loc[7] =  [107, "Computers, Internet, Digital Media & Technology"]
    df.loc[8] =  [108, "Cook Books"]
    df.loc[9] =  [109, "Dictionaries"]
    df.loc[10] = [110, "Drama"]
    df.loc[11] = [111, "Entertainment"]
    df.loc[12] = [112, "Engineering"]
    df.loc[13] = [113, "Educational"]
    df.loc[14] = [114, "Encyclopedia"]
    df.loc[15] = [115, "Fiction"]
    df.loc[16] = [116, "Guide"]
    df.loc[17] = [117, "Home, Crafts & Lifestyle"]
    df.loc[18] = [118, "Hobbies"]
    df.loc[19] = [119, "Horror"]    
    df.loc[20] = [120, "Health, Personal Development & Fitness"]
    df.loc[21] = [121, "History"]
    df.loc[22] = [122, "Humor & Satire"]
    df.loc[23] = [123, "Journals"]
    df.loc[24] = [124, "Law"]
    df.loc[25] = [125, "Language, Linguistics and Writing"] 
    df.loc[25] = [126, "Literature & Anthology"]        
    df.loc[26] = [127, "Medical"]
    df.loc[27] = [128, "Crime, Mystery, Thriller & Suspense"]
    df.loc[28] = [129, "Parenting/Relationship"]
    df.loc[29] = [130, "Politics & Social Sciences"]
    df.loc[30] = [131, "Poetry"]
    df.loc[31] = [132, "Reference"]
    df.loc[32] = [133, "Religions & Spirituality"]
    df.loc[33] = [134, "Romance"]
    df.loc[34] = [135, "Science & Maths"]
    df.loc[35] = [136, "Science Fiction & Fantasy"]
    df.loc[36] = [137, "Self Help"]
    df.loc[37] = [138, "Sports & Outdoors"]
    df.loc[38] = [139, "Teen & Young Adult"]
    df.loc[39] = [140, "Exam Preparation & Study tools"]
    df.loc[40] = [141, "Travel"]
    df.loc[41] = [142, "Adult wellness"]
    
    return df

def generate_genre_csv(path):
    df = generate_book_genre()
    df.to_csv(path);