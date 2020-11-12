# CS229-Project
## Description:
In this project, our group focuses on a self-crawled, highly inbalanced dataset on anime rates, using myAnimeList API, to represent data available to a new community. Our group carefully analyzes the challenge present for such dataset, and modifies the existing algorithms for recommendation system to apply these on such dataset.

## Structure:
This project includes several steps. You will be able to replicate the project by following the below commands. Notice, however, that due to the frequent modification of users information, if you crawled the data using our script, the detailed entries may subject to change. As a reference, a complete crawled data is included in the ```data``` folder

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

- step 6: Train models 
### Baseline and NN Models
Baseline and NN result is obtained if run with the below script. Note current NN model assumes there are 10 genres after step 4.
```
python3.7 nn.py 
```
### Item-Item Collaborative Filtering and K-Nearest Neighborhood Algorithm
Item-Item Collaborative Filtering and K-Nearest Neighborhood Algorithm can be run in
```
Milestone.ipynb
```
