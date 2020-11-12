# CS229-Project

- step 1: Parse anime data and user reviews from anime id 0 to anime id k
```
python3.7 parse_anime_info.py [k]
```

- step 2: Parse user data based on user names logged in step 1
```
python3.7 parse_user_info.py
```

- step 3: Parse anime data for animes within user's favorite anime list. Note k is added so we don't have duplicated entries for animes already parsed in step 1
```
python3.7 parse_anime_info.py fav_anime_list [k]
```

- step 4: Generate interested list of genres based on frequency threshold f
```
python3.7 extract_genres.py [f]
```

- step 5: Generate feature vector and split data into train, dev and test 
```
python3.7 split_data.py 
```

- step 6: Train NN model and make predictions. Note current NN model assumes there are 10 genres after step 4.
```
python3.7 nn.py 
```
