# data processing script for Neural Network approach

import pandas as pd
import numpy as np
import pickle

import random

anime_dataframe = pd.read_csv('cs229_anime_data.csv')
user_dataframe = pd.read_csv('cs229_user_data.csv')
review_dataframe = pd.read_csv('cs229_reviews_data.csv')

#genre_list
genre_list_file = 'cs229_genre_list.p'
genre_list = pickle.load(open(genre_list_file, "rb"))

# remove irrelevant information
adf = anime_dataframe.drop(columns=['Image Url', 'Title', 'Rank', 'Status'])
adf = adf.fillna(0)
anime_to_feature = {}

keys = adf.keys()

anime_genre_dict = {}
for _, anime in adf.iterrows():
    _id = anime['ID']
    #other animes are for fav anime list
    _genres = anime['Genres']
    genre_count = []
    for g in genre_list:
        if _genres != 0 and g in _genres:
            genre_count.append(1)
        else:
            genre_count.append(0)
    current_genre_list = np.array(genre_count)
    anime_genre_dict[_id] = current_genre_list
    slice1 = np.array(anime[:-2])
    slice2 = np.array(anime[-1:])
    anime_data_slice = np.concatenate((slice1,slice2))
    _feat = np.concatenate((anime_data_slice, current_genre_list))
    anime_to_feature[_id] = _feat

udf = user_dataframe.drop(columns=['Location'])
udf['Gender'] = udf['Gender'].fillna('None')
udf['Episodes_watched'] = udf['Episodes_watched'].fillna(0)
user_to_feature = {}

for _, user in udf.iterrows():
    _uname = user['Username']
    _feat = np.array(user[1:-1])
    _fav_list = user['Favorites_anime_id']
    if isinstance(_fav_list,str):
        _fav_list =_fav_list.split(',')
    else:
        _fav_list = []
    genre_count = np.zeros(len(genre_list))
    total = 0
    for id in _fav_list:
        if int(id) in anime_genre_dict:
            genre_count += anime_genre_dict[int(id)]
    user_to_feature[_uname] = np.concatenate((_feat, genre_count))

full_user_feat = []
full_anime_feat = []
full_rate = []

for _, review in review_dataframe.iterrows():
    _aid = review['Anime ID']
    _uname = review['Username']
    _rate = review['Scores_overall']
    if _uname not in user_to_feature or _aid not in anime_to_feature:
        continue
    full_user_feat.append(user_to_feature[_uname])
    full_anime_feat.append(anime_to_feature[_aid])
    full_rate.append(np.array([_rate]))

full_feat = np.concatenate((np.array(full_user_feat), np.array(full_anime_feat)), axis=1)
full_data = np.concatenate((np.array(full_feat), np.array(full_rate)), axis=1)

# shuffle the data
np.random.seed(110)
np.random.shuffle(full_data)
train_idx = (int)(full_data.shape[0] * 0.7)
val_idx = (int)(full_data.shape[0] * 0.9)
train_data, val_data, test_data = full_data[:train_idx, :], full_data[train_idx:val_idx, :], full_data[val_idx:, :]
print(train_data.shape)
print(val_data.shape)
print(test_data.shape)

column_values = ['Gender', 'Episodes Watched'] + genre_list + ['ID','Episodes', 'Rating', 'Average Score', 'Popularity', 'Members', 'Favorites', 'Adaption Size', 'Producers', 'Studios', 'Type'] + genre_list + ['Score']
out_dataframe = pd.DataFrame(data=train_data,columns=column_values)
out_dataframe.to_csv('train_data.csv',index=False)
out_dataframe = pd.DataFrame(data=val_data,columns=column_values)
out_dataframe.to_csv('val_data.csv',index=False)
out_dataframe = pd.DataFrame(data=test_data,columns=column_values)
out_dataframe.to_csv('test_data.csv',index=False)












