import pandas as pd
import pickle

output_folder = './'
genre_list_file = output_folder + 'cs229_genre_list.p'
threshold = 15
genre_list = []

def filter_genres():
    anime_dataframe = pd.read_csv('cs229_anime_data.csv')
    genre_dict = {}
    for index, row in anime_dataframe.iterrows():
        if row['ID'] >= 100:
            break
        temp_list = row['Genres'].split(',')
        print(temp_list)
        for genre in temp_list:
            if genre not in genre_dict:
                genre_dict[genre] = 1
            else:
                genre_dict[genre] += 1

    for genre,count in genre_dict.items():
        if count >= threshold:
            genre_list.append(genre)

    print(genre_dict)
    print(genre_list)
    pickle.dump(genre_list, open(genre_list_file, "wb"))


filter_genres()