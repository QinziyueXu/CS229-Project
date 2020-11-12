# -*- coding: utf-8 -*-
"""Milestone.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1w6UBekOUmh9cx4_Bc1RxeKHT6jtBqCM3

# Data Exploration and Analysis
Import thte .csv file crawled through myAnimeList APIs and import them
"""

from google.colab import files

uploaded = files.upload()

for fn in uploaded.keys():
  print('User uploaded file "{name}" with length {length} bytes'.format(
      name=fn, length=len(uploaded[fn])))

import pandas as pd
import numpy as np
import math
import copy

# load and convert to numpy array
anime_dataframe = pd.read_csv('cs229_anime_data.csv')
user_dataframe = pd.read_csv('cs229_user_data.csv')
review_dataframe = pd.read_csv('cs229_reviews_data.csv')
print("--------Displaying few rows in each data frame--------")

print("Anime data: ")
anime_dataframe.head(10)

print("User data: ")
user_dataframe.head(10)

print("Review data: ")
review_dataframe.head(10)

anime_data = anime_dataframe.to_numpy()
user_data = user_dataframe.to_numpy()
review_data = review_dataframe.to_numpy()
print("User Data Shape: %d, %d"%user_data.shape)
print("Anime Data Shape: %d, %d" %anime_data.shape)
print("Review Data Shape: %d, %d" %review_data.shape)

# create dictionary from user to id in order to create the
# interaction matrix
user2idx = {}
for i in range(len(user_data)):
    name = user_data[i,0]
    user2idx[name] = i
# create dictionary map real_anime_id to fake_anime_id
real2fake = {}
for i in range(len(anime_data)):
    real_id = anime_data[i, 0]
    real2fake[real_id] = i

# keep only the overall score for review
review_data = review_data[:,0:3]

# create the interaction matrix, missing rates are set to -1
interact_data = -np.ones((len(user_data),len(anime_data))).astype(int)

for i in range(len(review_data)):
    review = review_data[i, :]
    anime_idx = real2fake[review[0]]
    # in case user in the review not occured in user data
    if review[1] not in user2idx:
        continue
    user_idx = user2idx[review[1]]
    score = review[2]
    interact_data[user_idx, anime_idx] = score

print("User-Anime Interaction Matrix size: %d, %d"%interact_data.shape)

"""Decompose the anime genre data and transform it into a binary vectore representing if the anime is tagged with that genre.
User data is also transformed so missing entires will be filled as None
"""

# get all possible categories for an anime
genre2idx = {}
for i in range(len(anime_data)):
    genres = anime_data[i,-3].split(',')
    for genre in genres:
        if not genre in genre2idx:
            genre2idx[genre] = len(genre2idx)

genre_matrix = np.zeros((len(anime_data), len(genre2idx)))
for i in range(len(anime_data)):
    genres = anime_data[i,-3].split(',')
    for genre in genres:
        genre_idx = genre2idx[genre]
        genre_matrix[i, genre_idx] = 1

print("Total genres for Anime we possess: %d"%len(genre2idx))

# update user data so missing entries are replaced by None
# create id for each entry for user
user_data = np.concatenate(([[i] for i in range(len(user_data))], user_data), 1)
full_user_data = copy.deepcopy(user_data)
for user_info in full_user_data:
    for idx in range(len(user_info)):
        user_entry = user_info[idx]
        if isinstance(user_entry, float) and math.isnan(user_entry):
            user_info[idx] = 'None'

"""Writing the processed data into the storage."""

# write the interaction matrix
index_values = user_data[:, 1].reshape(-1,1)
column_values = anime_data[:,2]

out_dataframe = pd.DataFrame(data=interact_data, columns=column_values)
out_dataframe.insert(0,'User',index_values)

out_dataframe.to_csv('cs229_update_review.csv')

# overwrite genre info
full_anime_data = np.delete(anime_data,np.s_[-1:],axis=1)
full_anime_data = np.concatenate((anime_data,genre_matrix),1)

index_values = anime_data[:,2].reshape(-1, 1)
column_values = np.concatenate((['ID','Image Url','Title','Episodes','Rating','Score','Rank','Popularity','Members','Favorites','Adaption_Size','Producers','Studios','Genres','Type','Status'],np.array(list(genre2idx.keys()))))

out_dataframe = pd.DataFrame(data=full_anime_data,index=index_values,columns=column_values)
out_dataframe.to_csv('cs229_update_anime.csv')

# write user data with the replaced missing entries
column_values = ['Index','Username','Location','Gender','Episodes_watched','Favorites_anime_id']
out_dataframe = pd.DataFrame(data=full_user_data,columns=column_values)
out_dataframe.to_csv('cs229_update_user.csv')

"""# Exploration

## Review Data
To start with, we are going to use the original review data, i.e. the one not yet converted to interaction matrix, and explore the general distribution for the statistics we have.
"""

import matplotlib.pyplot as plt
average_score_per_anime = review_dataframe.groupby(by= 'Anime ID', as_index=False).agg({'Scores_overall':pd.Series.mean})
n, bins, patches = plt.hist(average_score_per_anime['Scores_overall'], 20, facecolor='lightseagreen',alpha=0.8)
plt.title('Distribution of Average Ratings per Anime in Dataset')
plt.xlabel('Average Score')
plt.ylabel('Frequency')
plt.show()
average_score_per_anime['Scores_overall'].describe()

"""Meanwhile, we also investigate if there is any of the 79 anime in this data set is extremely popular. For item like this, it might not mean too much to recommend such item to the user since they are already popular.
Fortunately, we only have a limited amount of anime in our data set that received more than 100 rates, which is 2.5% of total users we have. Hence, in this stage we choose not to consider about this issue yet, but in the future, we might think about removing these popular animes from the recommendation lists.
"""

total_score_per_anime = review_dataframe.groupby(by= 'Anime ID', as_index=False).agg({'Scores_overall':pd.Series.count})
n, bins, patches = plt.hist(total_score_per_anime['Scores_overall'], 20, facecolor='skyblue',alpha=0.8)
plt.title('Distribution of Average Rates per Anime Received in Dataset')
plt.xlabel('Rate Amounts')
plt.ylabel('Frequency')
plt.show()

print("Top 5 anime with most reviews")
total_score_per_anime.nsmallest(15, 'Scores_overall')
total_score_per_anime.mean()
total_score_per_anime.std()

average_score_per_user = review_dataframe.groupby(by= 'Username', as_index=False).agg({'Scores_overall':pd.Series.mean})
n, bins, patches = plt.hist(average_score_per_user['Scores_overall'], 20, facecolor='steelblue',alpha=0.8)
plt.title('Distribution of Average Ratings per User in Dataset')
plt.show()

"""We can see that the average rating per user screwe to the right strongly, which is reasonable since in general, people rate anime that find good with high score, holding the expectation to introduce the good one to more people.
This could potentially have a problem, as if a user rates all anime higher than 8, how could one measure if a recommendation is good or not, given that all animes he has rated in the past have a score of 8 and thus is in general considered as good.
In later section, we will discuss about this issue.

On the other hand, we will be able to see, with the below plot, that the data set holds little historic information about each individual user, since the majority of them rates only one of the 79 anime in the data set.
"""

total_score_per_user = review_dataframe.groupby(by= 'Username', as_index=False).agg({'Scores_overall':pd.Series.count})
n, bins, patches = plt.hist(total_score_per_user['Scores_overall'], 20, facecolor='maroon',alpha=0.8)
plt.title('Distribution of Number of Reviews per User')
plt.xlabel('Number of Rates')
plt.ylabel('Frequency')
plt.show()
total_score_per_user.describe()
print(len(total_score_per_user[total_score_per_user['Scores_overall'] == 1]))
print(len(total_score_per_user[total_score_per_user['Scores_overall'] == 2]))
print(len(total_score_per_user[total_score_per_user['Scores_overall'] == 3]))
print(len(total_score_per_user[total_score_per_user['Scores_overall'] == 4]))
print(len(total_score_per_user[total_score_per_user['Scores_overall'] >= 5]))
len(total_score_per_user)

"""Due to this restriction, later when we start our preliminary experiments on building up recommendation system, we will explain how this distribution results into the not-so-well scores givne by the user-user model and hence discuss our adjustment towards this issue.

## User Data
Since the data set is self-crawled, it is a relatively small data set and with certain amount of missing entries. 
In the below section, we will explore the weakness of this data set, and we will explore algorithms, hoepfully able to dig out algorithms that suit this type of data set, which can be common for start-up companies, where the data is limited and incomplete in general.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

anime_dataframe = pd.read_csv('cs229_update_anime.csv')
user_dataframe = pd.read_csv('cs229_update_user.csv')

anime_data = anime_dataframe.to_numpy()
user_data = user_dataframe.to_numpy()

complete_user = []
for user in user_data:
    complete = True
    for entry in user:
        if entry == 'None':
            complete = False
            break
    if complete:
        complete_user.append(user)

print("Number of user with no missing entry: %d"%len(complete_user))

"""We explore the missing entries for the users. Notice that majority of the users tend to fill out anime-related information while hold back about personal information. In later exploration, this information will be used and the intuition suggested that such information could reflect great features important in recommending anime to the user as it's more relevant."""

# count number of users with location information
with_location_user = []
for user in user_data:
    if user[2] == 'None':
        continue
    with_location_user.append(user)

print("Number of user with location information: %d"%len(with_location_user))

# count number of users with gender information
with_gender_user = []
for user in user_data:
    if user[3] == 'None':
        continue
    with_gender_user.append(user)

print("Number of user with gender information: %d"%len(with_gender_user))

# count number of users with favorite anime information
with_favorite_user = []
for user in user_data:
    if user[5] == 'None':
        continue
    with_favorite_user.append(user)

print("Number of user with favorite information: %d"%len(with_favorite_user))

"""The below analyze the neighborhood and overlap between users' rates. This serves as a supplementary analysis for the feasibility of the user-user collaborative filtering, which will be discussed a little in the report."""

# check user information available
full_review_dataframe = pd.read_csv('cs229_update_review.csv')
full_review_data = full_review_dataframe.to_numpy()

print("Rivew Data Shape: %d, %d"%full_review_data.shape)

user2review = {}
total_anime = len(anime_data)

for review in full_review_data:
    user = review[0]
    missing = np.count_nonzero(review[1:] == -1)
    user2review[user] = total_anime - missing

review2user = {}
for user in user2review:
    review_count = user2review[user]
    if review_count not in review2user:
        review2user[review_count] = 1
    else:
        review2user[review_count] += 1

print("User to Review amount")
print(user2review)
print("Reivew amount to User")
print(review2user)

# Analyze overlap between users
user2anime = {}
for review in full_review_data:
    user = review[1]
    for i in range(2, len(review)):
        if review[i] != -1:
            # if user reviewed this anime
            if user not in user2anime:
                user2anime[user] = set()
            user2anime[user].add(i)
    # if user not in user2anime:
    #     print("For some reason no review for %s"%user)

# compute overlap between user
overlap = {}
for user in user2anime:
    overlap[user] = {}
    for otherUser in user2anime:
        if user == otherUser:
            continue
        intersect = len(user2anime[user].intersection(user2anime[otherUser]))
        if intersect not in overlap[user]:
            overlap[user][intersect] = 0
        overlap[user][intersect] += 1

maxnb2count = {}
for user in overlap:
    maxnb = 0
    for count in overlap[user]:
        if count >= 3:
            maxnb = count
    if maxnb not in maxnb2count:
        maxnb2count[maxnb] = 0
    maxnb2count[maxnb] += 1
print("Max Valid Neighbhorhood Distribution")
print(maxnb2count)

overlap_count = {}
for user in user2anime:
    overlap_dict = overlap[user]
    for overlap_num in overlap_dict:
        if overlap_num not in overlap_count:
            overlap_count[overlap_num] = 0
        overlap_count[overlap_num] += 1
print("Overlap in General")
print(overlap_count)

"""# Experiments
As a primitive approach, we will attempt on how tradicitonal algorithm works on our data set, which is in many ways far away from perfect.
In later sections, we perform experiment on User-User collaborative Filter and Item-Item collaborative Filtering. For the first one, there are some changes to the algorithm in order to accomodate the fact that our data set suffers from the issue that many users have few and even no neighborhood that could be considered close.

## Compute Similarity Score
As a start, we start by computing similarity scores across all anime we have inside our data set, and with that we are going to analyze the implication of the similarity score and cast certain conjecture on performance of item-item collaborative filtering.
"""

def compute_similarity_matrix(reviews):
  """Take in reviews matrix of shape (number of user, number of anime)
  Return: matrix of shape (number of anime, number of anime), with matrix[i, j] = similarity(i, j)
  """
  # replace all -1 entries in reviews to nan
  for i in range(reviews.shape[0]):
    review = reviews[i, :]
    review[review == -1] = np.nan
    reviews[i,:] = review
  
  # normalize the score
  # we still keep users which only one rating and take faith in that
  # users with only one rating rates in absolute indifference
  # just to adapt to the situation that we have significant amount 
  # of users having only one rate
  normal_review = copy.deepcopy(reviews)
  # for i in range(reviews.shape[0]):
  #   review = np.array(reviews[i,:], dtype=float)
  #   # for each user, normalize user score
  #   avg_score = np.nanmean(review)
  #   max_min = np.nanmax(review) - np.nanmin(review)
  #   if max_min == 0:
  #     # if only have one score for everything, augment data
  #     # by putting dummy values (3 and 7)
  #     avg_score = np.nansum(review) + 3 + 7 / np.count_nonzero(~np.isnan(review))
  #     max_min = 4
  #   review = (review - avg_score) / max_min
  #   normal_review[i,:] = review

  # after normalization, convert missing values back
  normal_review = np.nan_to_num(normal_review, copy=True)

  # obtain score across all users
  review_square_matrix = np.power(normal_review, 2)
  score_square_sum = np.sum(review_square_matrix, axis=0)

  # adjust the review score matrix by the mean of each user
  # and reshape into column vector
  #user_avg_score = (np.sum(reviews, axis=1) / reviews.shape[1]).reshape(-1, 1)
  #reviews = reviews - user_avg_score
  score_prod_matrix = (normal_review.T).dot(normal_review)

  similarity_matrix = np.zeros(score_prod_matrix.shape)
  for i in range(score_prod_matrix.shape[0]):
    for j in range(score_prod_matrix.shape[1]):
      denominator = np.sqrt(score_square_sum[i]) * np.sqrt(score_square_sum[j])
      numerator = score_prod_matrix[i, j]
      similarity_matrix[i,j] = numerator / denominator
  
  return similarity_matrix

#convert review data to pure review matrix with no user information
review_matrix = copy.deepcopy(full_review_data[:,2:])

reviews = []
# drop people with no reviews
for review in review_matrix:
  has_score = False
  for score in review:
    if score != -1:
      has_score = True
      break
  if has_score:
    reviews.append(review)
reviews = np.array(reviews,dtype=float)

similarity_matrix = compute_similarity_matrix(reviews)

"""As we can see from the above output, the similarity score, due to the sparsity of the matrix, the similarity score is relatively low across all pair of anime. Due to the fact that most similarity score is low, it is suspicious whether the item-item collaborative filtering will be able to do a good job, measured by RMSE, which is the focus of the later section.

## Item-Item Collaborative Filtering
Unlike User-User Collaborative Filtering, Item-Item Collaborative Filtering has the advantage that it focuses on analyzing information surrounding the products (which, in this case, is the anime), which in general is far more accessible than the user information, and in particular for our data set.

Next, we are going to run experiment on item-item collaborative filtering. 
For each user, if the user never rates this item, we are going to ignore that item. Otherwise, we will use other anime that this user has seen before to compute the corresponding prediction.
"""

# compute values before
modified_review = np.zeros(reviews.shape)
for i in range(reviews.shape[0]):
  review = modified_review[i, :]
  review[review == -1] = np.nan
  modified_review[i,:] = review

avg_user_score = np.zeros((reviews.shape[0],1))
avg_item_score = np.zeros((reviews.shape[1],1))

for i in range(reviews.shape[0]):
  review = np.array(reviews[i, :],dtype=float)
  avg_user_score[i] = np.nanmean(review)
for i in range(reviews.shape[1]):
  review = np.array(reviews[:, i],dtype=float)
  avg_item_score[i] = np.nanmean(review)

# TODO: Modify so that only some (user, anime) tuples are removed from the training
# later those will be used as test case for accuracy. Current method does not work

def compute_score(similarity_matrix, reviews, avg_user_score, avg_item_score, user, anime):
  """Generate predicted scores for every user-item pair
  Args: similarity_matrix of shape (number of anime, number of anime)
        reviews: reviews matrix
        avg_*_score: average score computed with nan value removed
        user: idx of the user we want to predict
        anime: idx of the anime we want to predict
  Return: a real value as predicted value for this anime
  """
  # for each user, compute the score for each item
  avg_score = avg_item_score[anime]
  numerator = 0
  denominator = 0
  # loop over all possible other anime
  for j in range(similarity_matrix.shape[0]):
    # do not count itself
    if anime == j:
      continue
    # if user has not seen this before
    if np.isnan(reviews[user, j]):
      continue
    denominator += similarity_matrix[anime, j]
    #numerator += similarity_matrix[i,j] * (reviews[user, j] - avg_score)
    numerator += similarity_matrix[anime,j] * reviews[user, j]
  if denominator == 0:
    # if user has only seen one anime, discard
    pred_score = avg_score
  else:
    #pred_score = (numerator / denominator) + avg_user_score[user]
    pred_score = (numerator / denominator)
  return pred_score

# define metric function that determines recall and precision
#def compute_recall_precision(pred_y, y):

# extract indicies from randomly selected users and create the 
# training reviews from the selected users
similarity_matrix = compute_similarity_matrix(reviews)

# test on accuracy by going over the review entries and check 
# for difference
def compute_i2i_rmse(similarity_matrix, reviews, avg_user_score, avg_item_score):
  total_review = 0
  mse = 0
  for i in range(reviews.shape[0]):
    for j in range(reviews.shape[1]):
      actual_score = reviews[i, j]
      if np.isnan(actual_score):
        continue
      pred_score = compute_score(similarity_matrix, reviews, avg_user_score, avg_item_score, i, j)
      if pred_score == -1:
        continue
      #print(pred_score)
      se = (pred_score - actual_score)**2
      mse += se
      total_review += 1
  mse = mse / total_review
  return mse

print(compute_i2i_rmse(similarity_matrix, reviews, avg_user_score, avg_item_score))
#compute_baseline_rmse(reviews)

"""## User-User Collaborative Filtering
Next, we are going to run experiment on user-user collaborative filtering. For each user, we will construct a vector based on user's favorite anime liest, number of episodes watched and gender of the user.
"""

import math
import pandas as pd

#{username1: {male:1, 233:1, 496:1, episodes<4000:1}, username2: {female:1, 123:1, 456:1, episodes<2000:1}}
def load_user_info():
    user_info = {}
    user_dataframe = pd.read_csv('cs229_user_data.csv')

    for index, row in user_dataframe.iterrows():

        user_list = {}
        user_name = row['Username']
        gender = row['Gender']

        fav_anime_list = row['Favorites_anime_id']
        if isinstance(fav_anime_list,str):
            anime_list = fav_anime_list.split(',')
        else:
            anime_list = []

        user_list[gender] = 1
        for anime in anime_list:
            user_list[anime] = 1

        if row['Episodes_watched'] == 'None':
            key = 'episodesNA'
            user_list[key] = 1
        else:
            episodes_watched = float(row['Episodes_watched'])
            for i in range(10):
                if episodes_watched >= i * 2000 and episodes_watched < (i + 1) * 2000:
                    key = 'episodes<' + str((i + 1) * 2000)
                    user_list[key] = 1
            if episodes_watched > 20000:
                key = 'episodes>20000'
                user_list[key] = 1
        user_info[user_name] = user_list

    return user_info

"""Then we load review data which will be needed when make and evaluate predictions"""

#{anime1: {username1: score1, username2: score2}, anime2: {username1: score1, username2: score2} }
def load_review_info():
    review_dataframe = pd.read_csv('cs229_reviews_data.csv')

    review_matrix = {}
    for index, row in review_dataframe.iterrows():
        anime_id = row['Anime ID']
        username = row['Username']
        overall_score = float(row['Scores_overall'])
        if anime_id not in review_matrix:
            review_matrix[anime_id] = {}
        review_matrix[anime_id][username] = overall_score

    return review_matrix

def count_common(vec1, vec2):
    count = 0
    for k, v in vec1.items():
        if k in vec2:
            count += v * vec2[k]
    return count

"""Train function is needed to compute similarity between each pair of users."""

#{(username1,username2): similarity, (username2,username3): similarity,}
def train():
    similarity_matrix = {}
    user_info = load_user_info()

    for k1, v1 in user_info.items():
        for k2, v2 in user_info.items():
            if k1 == k2:
                continue
            if (k1,k2) in similarity_matrix or (k2,k1) in similarity_matrix:
                continue
            else:
                common = count_common(v1, v2)
                len_1 = len(v1)
                len_2 = len(v2)
                similarity = common / math.sqrt(len_1) / math.sqrt(len_2)
                similarity_matrix[(k1,k2)] = similarity
    return similarity_matrix

"""Predict functions finds k nearest neighbors of a given user from all users that have reviewed the anime and use the average score of the k users as the predicted score for the user. Leave one out average score is also calculated for baseline."""

def predict(similarity_matrix, review_truth, k_nearest):
    review_predict = {}
    avg_predict = {}
    for anime_id, user_review in review_truth.items():
        review_predict[anime_id] = {}
        avg_predict[anime_id] = {}
        for target, target_score in user_review.items():
            # find k nearest neighbor for target user
            k = k_nearest
            username_list = []
            user_similarity_list = []

            total_avg_score = 0
            avg_count = 0
            for others, others_score in user_review.items():
                if target == others:
                    continue
                total_avg_score += others_score
                avg_count += 1

                username_list.append(others)
                if (target,others) in similarity_matrix:
                    user_similarity_list.append(similarity_matrix[(target,others)])
                if (others, target) in similarity_matrix:
                    user_similarity_list.append(similarity_matrix[(others, target)])
            zip_list = zip(username_list, user_similarity_list)
            sorted_zip = sorted(zip_list, key=lambda x: x[1])

            user_results = sorted_zip[-k:]

            total_score = 0

            for user_tuple in user_results:
                username = user_tuple[0]
                total_score += user_review[username]

            if total_score != 0:
                total_score = total_score / len(user_results)

            #current review is the only review for the anime
            #assign a neutral value
            if avg_count == 0:
                total_score = 5
                total_avg_score = 5
                avg_count += 1

            review_predict[anime_id][target] = round(total_score)

            avg_predict[anime_id][target] = round((total_avg_score / avg_count))

    return review_predict, avg_predict

"""RMSE is calculated to measure the difference between predicted score and the actual review score provided by each user. F1 accuracy is calculated to show the performance for classification task. Results for average score method is also calculated as a baseline for experiment."""

def evaluate_result(prediction, truth, result_name):
    count = 0
    MSE = 0
    threshold = 7
    TP = 0
    TN = 0
    FP = 0
    FN = 0

    a = {}
    for i in range(10):
        a[i+1] = 0
    for k, v in prediction.items():

        for user, pred_score in v.items():
            truth_score = truth[k][user]
            a[int(truth_score)] += 1

            if pred_score >= threshold and truth_score >= threshold:
                TP += 1
            elif pred_score < threshold and truth_score < threshold:
                TN += 1
            elif pred_score < threshold and truth_score >= threshold:
                FN += 1
            elif pred_score >= threshold and truth_score < threshold:
                FP += 1
            count += 1
            MSE += (pred_score - truth_score) ** 2
    RMSE = math.sqrt(MSE / count)
    print(a)

    print("******* result for " + result_name + " *******")
    print("RMSE: " + str(RMSE))
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = 2 * (precision * recall) / (precision + recall)
    print("precision " + str(precision))
    print("recall " + str(recall))
    print("F1 " + str(F1))
    return RMSE


def main():
    similarity = train()
    review_truth = load_review_info()

    review_predict, avg_predict = predict(similarity, review_truth, 0)
    evaluate_result(avg_predict, review_truth, "average")

    best_k = 0
    best_rmse = 100
    for i in range(10):
        k_nearst = 5 + 5 * i
        review_predict, avg_predict = predict(similarity, review_truth, k_nearst)
        RMSE = evaluate_result(review_predict, review_truth, str(k_nearst))

        if best_rmse > RMSE:
            best_rmse = RMSE
            best_k = k_nearst

    print("Best RMSE is " + str(best_rmse) + " at k= " + str(best_k))


main()
