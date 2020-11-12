from jikanpy import Jikan
from jikanpy import exceptions
import pandas as pd
import time
import pickle
import csv

folder_name = 'results/'
csv_file_name = folder_name+"cs229_anime_data.csv"
csv_reviews_file_name = folder_name+"cs229_reviews_data.csv"
pickle_file_name = folder_name+"cs229_username_list.p"
#Actual list of anime can be found at:
#https://myanimelist.net/info.php?search=%25%25%25&go=relationids&divname=relationGen1
#The max id is 130285
start_id = 0
num_id_visited = 100

def extract_list_info(input_list, key):
    result = []
    for element in input_list:
        result.append(element[key])
    return ','.join(result)

output_folder = './results/'
genre_list_file = output_folder + 'cs229_genre_list.p'
genre_list = pickle.load(open(genre_list_file, "rb"))
print(genre_list)

def genre_extract_list_info(input_list, key):
    result = []
    for element in input_list:
        if element[key] in genre_list:
            result.append(element[key])
    return ','.join(result)

def main():

    fields = ['ID', 'Image Url', 'Title', 'Episodes', 'Rating', 'Score', 'Rank', 'Popularity', 'Members', 'Favorites', 'Adaption_Size', 'Producers', 'Studios', 'Genres', 'Type', 'Status']
    review_fields = ['Anime ID', 'Username', 'Scores_overall', 'Scores_story', 'Scores_animation', 'Scores_sound', 'Scores_character', 'Scores_enjoyment']
    jikan = Jikan()
    user_name_set = set()

    with open(csv_file_name, 'a') as csv_file:
        with open(csv_reviews_file_name, 'a') as csv_reviews_file:
            writer = csv.DictWriter(csv_file, fieldnames=fields)
            reviews_writer = csv.DictWriter(csv_reviews_file, fieldnames=review_fields)

            if csv_file.tell() == 0:
                writer.writeheader()
            if csv_reviews_file.tell() == 0:
                reviews_writer.writeheader()

            i = 0
            while i < num_id_visited:
                anime_id = start_id + i
                try:
                    #2 seconds per request
                    time.sleep(3)
                    #write anime info to csv file
                    anime = jikan.anime(anime_id)
                    content = {}
                    content['ID'] = anime_id
                    content['Image Url'] = anime['image_url']
                    content['Title'] = anime['title_english']
                    content['Episodes'] = anime['episodes']
                    content['Rating'] = anime['rating']
                    content['Score'] = anime['score']
                    content['Rank'] = anime['rank']
                    content['Popularity'] = anime['popularity']
                    content['Members'] = anime['members']
                    content['Favorites'] = anime['favorites']
                    if 'Adaptation' in anime['related']:
                        content['Adaption_Size'] = len(anime['related']['Adaptation'])
                    else:
                        content['Adaption_Size'] = 0

                    content['Producers'] = extract_list_info(anime['producers'], 'name')
                    content['Studios'] = extract_list_info(anime['studios'], 'name')
                    content['Genres'] = genre_extract_list_info(anime['genres'], 'name')

                    content['Type'] = anime['type']
                    content['Status'] = anime['status']

                    print(content)
                    writer.writerow(content)

                    #parse user info
                    for j in range(25):
                        time.sleep(2)
                        anime_reviews = jikan.anime(anime_id, extension='reviews', page = j + 1)
                        print("get " + str(len(anime_reviews['reviews'])) + " results")
                        for review in anime_reviews['reviews']:

                            review_info = {}
                            reviewer = review['reviewer']
                            user_name_set.add(reviewer['username'])
                            review_info['Anime ID'] = str(start_id + i)
                            review_info['Username'] = reviewer['username']
                            review_info['Scores_overall'] = reviewer['scores']['overall']
                            review_info['Scores_story'] = reviewer['scores']['story']
                            review_info['Scores_animation'] = reviewer['scores']['animation']
                            review_info['Scores_sound'] = reviewer['scores']['sound']
                            review_info['Scores_character'] = reviewer['scores']['character']
                            review_info['Scores_enjoyment'] = reviewer['scores']['enjoyment']
                            reviews_writer.writerow(review_info)

                        if len(anime_reviews['reviews']) < 20:
                            break

                    i += 1

                except exceptions.APIException:
                    print("anime id " + str(anime_id) + " does not exist")
                    i += 1
                except:
                    print("failure at " + str(anime_id))
                    print("Have collected " + str(len(user_name_set)) + " usernames")
                    pickle.dump(user_name_set, open(pickle_file_name, "wb"))
                    raise

    pickle.dump(user_name_set, open(pickle_file_name, "wb"))
    print("Have collected " + str(len(user_name_set)) + " usernames")

def parse_no_user_info(list_to_parse):
    fields = ['ID', 'Image Url', 'Title', 'Episodes', 'Rating', 'Score', 'Rank', 'Popularity', 'Members', 'Favorites',
              'Adaption_Size', 'Producers', 'Studios', 'Genres', 'Type', 'Status']
    jikan = Jikan()


    with open(csv_file_name, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)

        if csv_file.tell() == 0:
            writer.writeheader()

        for anime_id in list_to_parse:
            try:
                # 2 seconds per request
                time.sleep(2)
                # write anime info to csv file
                anime = jikan.anime(anime_id)
                content = {}
                content['ID'] = anime_id
                content['Image Url'] = anime['image_url']
                content['Title'] = anime['title_english']
                content['Episodes'] = anime['episodes']
                content['Rating'] = anime['rating']
                content['Score'] = anime['score']
                content['Rank'] = anime['rank']
                content['Popularity'] = anime['popularity']
                content['Members'] = anime['members']
                content['Favorites'] = anime['favorites']
                if 'Adaptation' in anime['related']:
                    content['Adaption_Size'] = len(anime['related']['Adaptation'])
                else:
                    content['Adaption_Size'] = 0

                content['Producers'] = extract_list_info(anime['producers'], 'name')
                content['Studios'] = extract_list_info(anime['studios'], 'name')
                content['Genres'] = genre_extract_list_info(anime['genres'], 'name')

                content['Type'] = anime['type']
                content['Status'] = anime['status']

                print(content)
                writer.writerow(content)
            except exceptions.APIException:
                print("anime id " + str(anime_id) + " does not exist")
            except:
                print("failure at " + str(anime_id))
                raise

def parse_fav_list():
    result = set()
    user_dataframe = pd.read_csv('results/cs229_user_data.csv')

    for index, row in user_dataframe.iterrows():
        if isinstance(row['Favorites_anime_id'], str):
            temp_list = row['Favorites_anime_id'].split(',')
            for id in temp_list:
                result.add(int(id))
    print(result)
    return result


if __name__ == '__main__':
    #main()

    anime_list = []
    anime_list.extend(parse_fav_list())
    for i in range(100):
        if i in anime_list:
            anime_list.remove(i)
    print(anime_list)
    parse_no_user_info(anime_list)





