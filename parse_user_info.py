from jikanpy import Jikan
from jikanpy import exceptions
import pickle
import time
import csv
from os import path

import parse_anime_info


usernames = pickle.load(open(parse_anime_info.pickle_file_name, "rb"))
folder_name = 'results/'
csv_file_name = folder_name+"cs229_user_data.csv"

#if there is an unexpected exception, we don't want to parse repetitive data
parsed_dict_location = folder_name+"parsed.p" #any user stored at here will not be parsed again
parsed_dict_previous = {}
if path.exists(parsed_dict_location):
    parsed_dict_previous = pickle.load(open(parsed_dict_location, "rb"))
    print(parsed_dict_previous)
    print(len(parsed_dict_previous))

parsed_dict = {}

def extract_list_info(input_list, key):
    result = []
    for element in input_list:
        result.append(str(element[key]))
    return ','.join(result)

def main():

    fields = ['Username', 'Location', 'Gender', 'Episodes_watched', 'Favorites_anime_id']
    jikan = Jikan()


    with open(csv_file_name, 'a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fields)

        if csv_file.tell() == 0:
            writer.writeheader()

        for username in usernames:
            if username in parsed_dict_previous or username in parsed_dict:
                continue
            try:
                #2 seconds per request
                time.sleep(2)
                #write anime info to csv file
                user = jikan.user(username)

                content = {}
                content['Username'] = username
                content['Location'] = user['location']
                content['Gender'] = user['gender']
                content['Episodes_watched'] = user['anime_stats']['episodes_watched']
                content['Favorites_anime_id'] = extract_list_info(user['favorites']['anime'], 'mal_id')

                print(content)
                writer.writerow(content)
                parsed_dict[username] = 1

            except exceptions.APIException:
                print("Username: " + str(username) + " does not exist")
                parsed_dict[username] = 1
            except:
                print("failure at " + str(username))
                for key, val in parsed_dict_previous.items():
                    parsed_dict[key] = val
                pickle.dump(parsed_dict, open(parsed_dict_location, "wb"))
                raise
    for key, val in parsed_dict_previous.items():
        parsed_dict[key] = val
    pickle.dump(parsed_dict, open(parsed_dict_location, "wb"))


if __name__ == '__main__':
    main()

