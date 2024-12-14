import streamlit as st
import pandas as pd
from pandas import NA
from itertools import combinations
import math
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from star_ratings import star_ratings
pd.options.mode.chained_assignment = None  # Disable SettingWithCopyWarning

from IPython.display import HTML

# URLs for datasets and images
movies_url = 'https://liangfgithub.github.io/MovieData/movies.dat?raw=true'
ratings_url = 'https://liangfgithub.github.io/MovieData/ratings.dat?raw=true'
users_url = 'https://liangfgithub.github.io/MovieData/users.dat?raw=true'
image_url_base = 'https://liangfgithub.github.io/MovieImages/'

# # Define column names
# user_cols = ['UserID', 'Gender', 'Age', 'Occupation', 'Zip-code']
# movie_cols = ['MovieID', 'Title', 'Genres']
# rating_cols = ['UserID', 'MovieID', 'Rating', 'Timestamp']

# # Load datasets from URLs with encoding to resolve Unicode errors
# users = pd.read_csv(users_url, sep='::', engine='python', header=None, names=user_cols, encoding='ISO-8859-1')
# movies = pd.read_csv(movies_url, sep='::', engine='python', header=None, names=movie_cols, encoding='ISO-8859-1')
# ratings = pd.read_csv(ratings_url, sep='::', engine='python', header=None, names=rating_cols, encoding='ISO-8859-1')

# # 1. Merge ratings with movie titles
# ratings_with_movies = ratings.merge(movies, on='MovieID')

# # 2. Define popularity as movies with the highest number of ratings
# movie_popularity = ratings_with_movies.groupby(['MovieID', 'Title'])['Rating'].agg(['count']).reset_index()
# movie_popularity.rename(columns={'count': 'NumRatings'}, inplace=True)

# # 3. Filter movies with a significant number of ratings (e.g., > 100)
# popular_movies = movie_popularity[movie_popularity['NumRatings'] > 100]

# # 4. Sort by NumRatings
# popular_movies = popular_movies.sort_values(by=['NumRatings'], ascending=[False]).reset_index(drop=True)

# # 5. Display the top 10 movies
# top_10_movies = popular_movies.head(10)

# # 6. Add movie poster URLs for Top 10 Movies
# top_10_movies['Poster'] = top_10_movies['MovieID'].apply(lambda x: f"{image_url_base}{x}.jpg")

# # Print the results
# #print("Top 10 Popular Movies Based on Number of Ratings")
# #print(top_10_movies[['MovieID', 'Title', 'NumRatings', 'Poster']])

# # Save popular movies to a CSV file for use in System II
# popular_movies.to_csv('popular_movies.csv', index=False)
# print("Popular movies saved as 'popular_movies.csv'")

# # Save movie information (MovieID, Title) to a CSV file for use in System II
# movies[['MovieID', 'Title']].to_csv('movie_info.csv', index=False)
# print("Movie information saved as 'movie_info.csv'")


def generate_similarity_matrix() :
    rating_matrix = pd.read_csv('ratings_matrix.csv')
    row_means = rating_matrix.mean(axis=1, skipna=True)
    rating_matrix_normalized = rating_matrix.sub(row_means, axis=0)

    B = rating_matrix_normalized.notna().astype(int)
    intersection_counts = B.T.dot(B)
    X = rating_matrix_normalized.fillna(0).values
    numerator_matrix = X.T.dot(X)
    norms = np.sqrt((X**2).sum(axis=0))
    denom_matrix = np.outer(norms, norms)
    cos_sim_matrix = numerator_matrix / denom_matrix
    sim_matrix = 0.5 + 0.5 * cos_sim_matrix
    similarity_matrix = pd.DataFrame(data=sim_matrix, index=rating_matrix_normalized.columns, columns=rating_matrix_normalized.columns, dtype=object)
    similarity_matrix[intersection_counts < 3] = pd.NA
    similarity_matrix[denom_matrix == 0] = pd.NA
    np.fill_diagonal(similarity_matrix.values, pd.NA)

    similarity_matrix.to_csv('similarity_matrix.csv')



def myIBCF(newuser):
    generate_similarity_matrix()
    # Load similarity matrix
    S = pd.read_csv('similarity_matrix.csv', index_col=0)
    movie_names = S.columns.values
    
    # Convert newuser to a NumPy array of floats, ensuring pd.NA -> np.nan
    newuser = pd.Series(newuser)            # ensure it's a Series
    newuser = newuser.replace({pd.NA: np.nan})
    newuser = newuser.astype(float).values  # now newuser is a float NumPy array with np.nan for missing values
    
    user_rated_mask = ~np.isnan(newuser)
    user_unrated_mask = np.isnan(newuser)
    predicted_ratings = np.full(S.shape[0], np.nan)
    user_unrated_indices = np.where(user_unrated_mask)[0]
    
    for i in user_unrated_indices:
        row = S.iloc[i]  # Pandas Series for row i
        valid_sim_mask = row.notna().values
        valid_neighbor_indices = np.where(valid_sim_mask & user_rated_mask)[0]
        
        if len(valid_neighbor_indices) == 0:
            continue
        
        sims = row.values[valid_neighbor_indices]
        neighbor_ratings = newuser[valid_neighbor_indices]
        
        denom = np.nansum(sims)
        if denom == 0:
            continue
        
        numerator = np.nansum(sims * neighbor_ratings)
        predicted_rating = numerator / denom
        predicted_ratings[i] = predicted_rating
    
    valid_predictions_mask = ~np.isnan(predicted_ratings)
    candidates = np.where(valid_predictions_mask & user_unrated_mask)[0]
    
    
    k = min(10, len(candidates))
    top_indices = candidates[np.argsort(predicted_ratings[candidates])[-k:]]
    top_indices = top_indices[np.argsort(predicted_ratings[top_indices])[::-1]]

    recommended_movies = movie_names[top_indices]
    for i in range(len(recommended_movies)):
        recommended_movies[i] = recommended_movies[i][1:]

    df = pd.read_csv('popular_movies.csv')

    for _, row in df.iterrows():
        movie_id = row['MovieID']
        if movie_id not in recommended_movies:
            np.append(recommended_movies, movie_id)        
        if len(recommended_movies) == 10:
            break

    recommended_movies = [int(item) for item in recommended_movies]
    df2 = pd.read_csv('movie_info.csv')
    matched_movies = df2[df2['MovieID'].isin(recommended_movies)]

    print(matched_movies[['Title']])
    return matched_movies[['Title']]


def recommend_movies(user_ratings):
    u1181_ratings = [3,4,NA,NA,NA,NA,3,NA,2,2,2,NA,NA,NA,NA,NA,3,NA,NA,3,2,2,2,NA,3,NA,3,NA,NA,NA,NA,2,3,NA,NA,NA,NA,NA,NA,1,3,5,NA,NA,NA,3,NA,NA,2,NA,NA,NA,NA,3,2,4,3,NA,3,NA,NA,4,NA,NA,NA,2,3,NA,NA,NA,3,NA,NA,3,NA,NA,NA,NA,NA,4,3,2,NA,4,2,3,NA,5,NA,3,NA,3,3,4,2,3,3,3,3,2,2,NA,NA,4,3,2,3,NA,2,3,NA,NA,5,NA,NA,3,NA,NA,NA,NA,NA,NA,4,NA,3,NA,3,3,NA,4,2,2,2,2,NA,3,NA,NA,2,4,NA,NA,NA,NA,NA,NA,3,4,NA,NA,NA,3,NA,2,3,NA,3,NA,NA,NA,NA,NA,NA,NA,NA,2,2,3,2,NA,NA,3,NA,4,3,NA,NA,2,3,NA,3,3,2,3,3,NA,3,3,2,3,NA,4,4,3,4,NA,NA,4,3,NA,2,5,NA,4,4,4,5,NA,3,3,3,3,4,4,NA,3,3,4,NA,3,5,4,3,3,3,3,3,3,NA,3,3,NA,5,3,NA,NA,4,3,NA,5,NA,1,3,3,2,2,2,3,4,NA,4,3,5,NA,4,NA,3,NA,4,3,NA,4,4,4,3,3,4,4,4,2,4,NA,3,2,5,2,4,2,4,NA,3,2,NA,NA,3,3,4,5,3,2,4,4,2,NA,NA,4,3,3,3,NA,2,3,4,3,NA,NA,3,2,3,3,4,2,NA,3,NA,NA,NA,NA,NA,NA,NA,NA,NA,3,3,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,3,NA,NA,NA,NA,2,NA,NA,NA,NA,3,4,3,3,NA,3,NA,3,3,NA,NA,NA,NA,NA,3,NA,3,3,NA,NA,NA,3,NA,NA,NA,2,3,NA,NA,4,2,4,3,4,3,3,2,2,2,NA,2,NA,3,NA,NA,4,NA,3,1,NA,NA,3,4,NA,3,2,3,2,NA,NA,1,NA,NA,3,2,3,NA,2,2,NA,2,2,3,NA,3,NA,NA,3,NA,3,NA,NA,NA,2,NA,NA,NA,NA,NA,3,NA,NA,NA,NA,NA,NA,NA,3,2,2,NA,NA,2,NA,2,NA,NA,3,2,NA,NA,3,NA,NA,2,NA,3,2,NA,1,NA,NA,NA,3,2,NA,NA,NA,NA,NA,1,NA,2,NA,3,NA,NA,2,4,NA,2,NA,NA,1,NA,NA,NA,NA,NA,NA,NA,2,NA,2,NA,3,3,NA,NA,1,NA,NA,NA,NA,2,NA,4,3,NA,2,NA,4,1,NA,NA,NA,NA,NA,NA,NA,5,NA,NA,2,NA,NA,NA,NA,NA,3,NA,NA,3,NA,NA,NA,2,NA,NA,NA,NA,NA,NA,NA,NA,3,NA,NA,NA,1,NA,4,NA,1,NA,NA,NA,2,NA,3,4,NA,NA,NA,3,NA,NA,NA,NA,4,NA,NA,NA,2,NA,3,3,NA,NA,4,2,3,NA,2,3,NA,NA,4,NA,NA,3,1,NA,NA,NA,2,NA,NA,1,3,NA,3,3,2,NA,2,2,NA,4,4,NA,NA,2,NA,NA,NA,NA,4,4,NA,NA,NA,NA,4,NA,3,3,NA,3,3,NA,3,NA,NA,1,3,NA,NA,NA,4,4,NA,NA,NA,4,NA,NA,NA,NA,NA,NA,NA,NA,3,NA,3,NA,3,NA,NA,NA,2,NA,NA,NA,NA,3,3,3,3,2,2,NA,3,3,1,2,NA,NA,NA,NA,4,NA,2,NA,5,NA,3,3,NA,NA,NA,3,2,2,3,NA,4,NA,NA,NA,3,2,NA,NA,NA,NA,4,3,NA,2,4,NA,NA,5,NA,NA,3,2,3,NA,2,NA,NA,1,NA,NA,NA,NA,NA,2,3,4,NA,NA,NA,NA,NA,3,NA,NA,NA,NA,NA,NA,2,NA,NA,4,NA,NA,NA,NA,NA,3,2,NA,NA,NA,2,3,NA,NA,NA,NA,NA,2,NA,2,NA,NA,2,NA,NA,NA,3,NA,3,2,NA,NA,NA,NA,3,NA,NA,NA,NA,NA,NA,NA,NA,2,NA,NA,NA,NA,2,NA,NA,NA,2,NA,NA,4,NA,2,NA,NA,NA,3,3,NA,NA,NA,NA,NA,3,1,2,NA,NA,NA,NA,NA,NA,NA,NA,3,NA,NA,NA,NA,2,1,NA,NA,2,NA,NA,1,NA,3,NA,NA,NA,NA,NA,4,NA,NA,NA,NA,NA,NA,3,3,NA,3,NA,NA,2,NA,NA,NA,NA,2,NA,NA,NA,NA,3,NA,NA,2,4,NA,NA,NA,NA,NA,NA,NA,NA,NA,3,NA,NA,NA,NA,2,NA,NA,NA,3,NA,NA,NA,2,3,NA,4,2,NA,NA,5,NA,3,3,NA,4,NA,4,NA,4,3,NA,NA,3,4,3,3,4,NA,4,2,3,2,NA,4,2,3,NA,3,3,3,3,3,3,3,3,3,3,4,NA,3,3,3,3,NA,NA,NA,NA,NA,NA,NA,NA,3,NA,NA,NA,NA,NA,NA,NA,NA,2,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,3,NA,NA,3,1,NA,1,2,NA,2,2,2,3,2,NA,5,NA,NA,3,NA,3,2,2,2,NA,3,1,3,NA,4,NA,3,2,2,4,NA,3,NA,NA,4,NA,2,NA,NA,NA,NA,NA,NA,NA,NA,NA,4,2,NA,NA,NA,2,NA,2,NA,3,NA,NA,NA,NA,NA,2,3,NA,NA,3,3,NA,3,NA,NA,2,NA,3,NA,4,3,2,2,NA,3,NA,NA,NA,4,3,2,2,3,NA,2,NA,1,NA,NA,NA,3,NA,NA,3,3,2,2,NA,NA,NA,3,NA,NA,NA,NA,2,3,3,3,NA,NA,NA,3,3,3,NA,4,2,NA,3,NA,3,NA,2,3,NA,3,NA,NA,NA,NA,NA,NA,NA,4,3,2,NA,NA,NA,NA,3,2,NA,3,3,3,3,NA,2,3,4,2,2,NA,2,3,3,2,NA,NA,NA,4,3,2,NA,1,NA,2,NA,NA,NA,4,NA,3,NA,NA,3,3,NA,NA,2,NA,NA,2,3,4,NA,2,1,NA,3,NA,3,NA,NA,1,3,3,2,NA,NA,3,NA,NA,NA,NA,NA,NA,NA,3,4,NA,2,NA,NA,3,NA,NA,NA,2,3,3,NA,4,NA,3,NA,3,2,3,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,3,2,3,NA,NA,NA,NA,NA,NA,3,3,3,2,NA,NA,3,NA,3,NA,2,3,4,2,NA,NA,2,1,NA,2,NA,NA,NA,2,NA,3,NA,2,NA,2,NA,NA,NA,3,2,2,NA,NA,3,NA,NA,NA,5,NA,NA,NA,2,3,3,NA,NA,NA,NA,2,4,NA,2,3,NA,NA,2,4,NA,2,NA,NA,2,NA,4,2,4,4,NA,NA,NA,NA,NA,NA,1,NA,NA,NA,3,NA,4,2,NA,3,3,NA,3,4,NA,3,NA,NA,NA,NA,4,3,NA,2,NA,NA,4,4,3,NA,NA,NA,2,NA,NA,NA,NA,3,3,2,NA,3,3,3,NA,4,2,3,NA,3,NA,NA,NA,3,NA,NA,NA,NA,NA,NA,NA,3,NA,NA,NA,NA,NA,3,NA,2,2,1,3,NA,3,NA,NA,2,NA,NA,NA,NA,2,NA,NA,NA,NA,NA,NA,3,NA,3,2,3,3,NA,5,NA,2,NA,4,3,2,2,2,3,3,2,NA,2,NA,2,2,3,2,3,NA,NA,2,1,NA,NA,3,2,2,NA,3,2,NA,2,1,NA,NA,NA,2,NA,3,NA,NA,NA,NA,NA,3,NA,3,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,1,NA,NA,NA,2,3,NA,2,3,NA,3,1,NA,NA,2,NA,NA,2,NA,2,3,3,4,2,3,2,3,NA,3,NA,4,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,2,NA,NA,NA,2,2,1,NA,NA,2,NA,NA,NA,NA,NA,NA,1,NA,NA,NA,3,NA,3,NA,NA,NA,NA,NA,NA,3,NA,2,2,NA,NA,NA,2,2,2,2,2,3,2,2,2,NA,NA,NA,2,NA,NA,NA,3,NA,4,1,3,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,2,2,4,NA,3,NA,NA,2,NA,1,NA,5,2,NA,NA,NA,NA,NA,NA,NA,NA,NA,3,NA,3,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,3,3,NA,4,4,NA,3,NA,NA,NA,NA,3,NA,NA,3,2,NA,3,2,4,NA,NA,NA,NA,NA,3,NA,2,4,NA,NA,4,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,3,2,1,2,NA,NA,NA,NA,3,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,2,NA,NA,3,NA,4,NA,NA,NA,NA,NA,4,3,3,3,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,3,NA,3,NA,2,NA,NA,2,NA,4,NA,NA,NA,NA,NA,2,NA,NA,4,1,2,NA,NA,NA,5,2,NA,NA,NA,2,NA,4,3,3,NA,4,2,2,1,3,1,NA,3,4,NA,NA,2,NA,4,4,2,2,3,3,NA,NA,3,NA,1,2,1,NA,NA,2,NA,NA,NA,4,3,4,NA,NA,NA,2,NA,1,NA,NA,NA,NA,3,NA,3,2,NA,NA,4,3,3,NA,NA,NA,NA,NA,2,2,NA,NA,NA,NA,NA,2,NA,NA,4,NA,NA,3,NA,NA,NA,NA,NA,NA,4,2,NA,NA,4,2,NA,2,2,NA,3,NA,NA,3,NA,NA,NA,2,2,NA,2,NA,1,2,NA,NA,4,NA,NA,NA,NA,2,1,1,1,3,2,NA,NA,NA,NA,NA,NA,4,2,NA,2,NA,NA,NA,NA,NA,NA,2,NA,NA,3,3,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,3,NA,NA,NA,NA,3,3,4,NA,2,3,2,3,NA,NA,3,NA,NA,NA,NA,NA,3,2,NA,NA,NA,NA,2,NA,NA,5,NA,2,NA,NA,NA,NA,NA,NA,NA,3,2,3,4,NA,NA,NA,NA,NA,4,NA,NA,NA,3,NA,NA,2,NA,NA,4,1,NA,NA,NA,NA,NA,5,NA,NA,3,4,2,5,3,2,3,3,3,NA,4,4,2,3,NA,3,5,NA,NA,3,NA,NA,NA,5,NA,NA,NA,NA,3,3,3,NA,3,2,4,4,4,4,NA,2,3,3,NA,NA,2,5,4,NA,NA,NA,NA,NA,NA,3,NA,3,NA,NA,3,3,NA,5,NA,3,2,1,NA,NA,3,NA,NA,3,NA,4,2,3,4,3,2,3,3,NA,3,NA,1,NA,2,NA,NA,1,NA,3,NA,NA,NA,NA,NA,3,3,2,NA,3,NA,NA,NA,NA,2,2,NA,NA,3,2,NA,4,NA,4,1,NA,NA,NA,3,2,4,4,NA,NA,3,NA,NA,1,4,NA,3,NA,3,NA,NA,NA,3,NA,NA,NA,NA,NA,2,NA,NA,3,3,NA,NA,NA,NA,NA,3,2,4,3,NA,NA,NA,1,3,3,NA,5,4,3,3,NA,3,NA,NA,NA,NA,3,5,2,3,NA,NA,NA,NA,2,4,2,NA,4,3,NA,5,NA,4,NA,3,3,NA,NA,NA,4,3,2,NA,2,2,NA,NA,3,2,NA,3,3,NA,2,3,NA,NA,4,NA,NA,NA,NA,NA,3,NA,NA,4,NA,NA,3,4,NA,NA,NA,NA,NA,5,3,NA,NA,3,NA,3,NA,2,NA,NA,NA,2,NA,3,NA,NA,2,NA,NA,3,NA,NA,4,NA,NA,NA,NA,2,5,NA,NA,NA,NA,2,NA,NA,4,3,3,NA,NA,3,NA,4,3,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,4,NA,NA,NA,3,4,3,3,NA,4,NA,NA,3,NA,NA,2,NA,NA,NA,NA,NA,NA,3,2,NA,4,1,NA,NA,3,NA,3,NA,NA,2,3,NA,3,NA,NA,NA,3,NA,NA,NA,NA,NA,NA,NA,NA,NA,3,NA,3,2,NA,3,3,2,1,1,NA,2,NA,2,1,3,3,2,NA,3,NA,2,NA,1,3,3,3,NA,2,NA,2,NA,NA,3,3,NA,3,4,NA,NA,NA,NA,NA,NA,NA,NA,2,NA,NA,NA,NA,4,NA,NA,NA,NA,NA,NA,4,NA,3,NA,NA,NA,3,3,NA,NA,NA,NA,NA,NA,3,NA,NA,NA,3,NA,NA,NA,NA,3,3,NA,NA,NA,1,NA,NA,NA,NA,2,3,4,5,2,NA,NA,NA,NA,3,NA,NA,NA,NA,2,4,NA,NA,2,NA,NA,NA,NA,3,NA,NA,3,NA,NA,NA,4,4,NA,2,3,NA,4,3,3,3,NA,3,3,NA,2,NA,3,3,NA,NA,NA,NA,NA,NA,NA,3,1,NA,NA,NA,NA,3,3,4,NA,NA,2,NA,NA,NA,NA,NA,NA,2,NA,NA,NA,4,3,NA,NA,NA,NA,NA,NA,NA,NA,3,4,NA,NA,1,NA,NA,NA,3,NA,NA,NA,3,2,3,NA,3,3,3,3,3,NA,NA,NA,3,1,2,1,NA,5,NA,NA,4,NA,3,NA,2,NA,NA,5,3,4,NA,3,2,4,3,3,3,NA,NA,NA,3,NA,NA,NA,NA,NA,NA,NA,NA,2,NA,3,5,NA,NA,NA,3,NA,NA,NA,NA,NA,NA,3,NA,NA,NA,4,NA,NA,NA,NA,NA,NA,NA,2,4,NA,NA,NA,NA,3,NA,2,2,3,NA,NA,3,2,NA,NA,NA,3,1,NA,3,4,2,NA,4,NA,NA,3,NA,NA,3,NA,NA,NA,NA,NA,NA,NA,NA,3,2,NA,4,NA,3,3,NA,NA,NA,NA,NA,3,3,NA,NA,NA,3,NA,NA,3,NA,4,3,NA,NA,4,NA,1,3,3,NA,NA,3,NA,1,NA,3,NA,NA,NA,NA,2,NA,NA,NA,4,NA,NA,NA,NA,NA,NA,NA,1,4,NA,NA,NA,NA,2,NA,NA,2,NA,NA,NA,2,NA,3,NA,NA,1,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,2,NA,NA,NA,NA,2,NA,NA,NA,4,2,3,NA,NA,3,3,4,NA,NA,NA,NA,2,NA,NA,NA,2,4,3,NA,NA,2,2,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,4,2,NA,NA,NA,4,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,3,NA,4,NA,1,NA,NA,NA,3,2,2,NA,NA,2,NA,4,3,1,3,3,2,NA,2,NA,1,NA,NA,NA,NA,NA,NA,NA,3,4,3,NA,3,3,2,4,3,3,1,3,1,NA,NA,2,3,NA,NA,NA,NA,NA,NA,NA,2,NA,2,3,NA,NA,3,NA,NA,NA,NA,3,3,3,4,NA,NA,3,3,4,5,NA,3,NA,2,3,4,4,NA,3,3,NA,NA,NA,4,NA,2,NA,3,NA,3,NA,NA,NA,NA,3,NA,NA,NA,3,2,NA,1,1,1,NA,4,2,2,NA,4,2,NA,NA,NA,NA,NA,NA,NA,NA,NA,4,NA,NA,NA,NA,3,NA,3,NA,2,2,4,NA,NA,NA,NA,NA,NA,NA,3,NA,3,2,NA,NA,2,NA,NA,NA,2,NA,NA,4,2,2,3,NA,NA,NA,3,3,NA,NA,NA,NA,1,2,2,3,NA,3,NA,NA,NA,NA,2,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,2,NA,NA,NA,NA,NA,NA,NA,NA,4,NA,NA,NA,NA,NA,NA,NA,NA,3,NA,3,NA,NA,NA,NA,4,3,NA,NA,3,NA,2,NA,NA,NA,1,NA,2,NA,NA,NA,3,NA,NA,NA,NA,NA,NA,2,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,3,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,4,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,2,NA,2,NA,3,NA,NA,2,3,NA,4,NA,NA,NA,3,2,NA,NA,NA,NA,NA,NA,4,NA,NA,3,3,3,4,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,1,NA,1,NA,NA,4,2,NA,4,3,NA,2,NA,5,NA,2,2,2,3,NA,2,2,NA,NA,NA,4,NA,NA,NA,NA,2,NA,3,3,1,NA,NA,NA,2,4,3,2,NA,NA,2,NA,NA,NA,NA,3,NA,NA,NA,3,3,2,NA,3,NA,NA,2,NA,NA,NA,NA,3,2,2,NA,3,NA,3,NA,1,1,NA,3,3,NA,NA,NA,NA,NA,2,3,NA,NA,NA,4,NA,2,NA,1,NA,3,2,2,2,5,3,NA,NA,NA,NA,NA,4,NA,NA,1,NA,NA,NA,NA,3,NA,NA,NA,3,NA,NA,2,NA,3,3,NA,1,NA,2,2,3,NA,3,NA,NA,3,NA,3,NA,NA,NA,3,2,2,4,NA,3,3,3,NA,1,4,NA,NA,2,NA,NA,NA,2,3,NA,NA,3,3,3,NA,NA,NA,NA,NA,NA,NA,NA,2,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,3,NA,NA,NA,NA,NA,3,2,3,2,5,NA,2,NA,3,3,3,2,NA,3,NA,5,3,2,NA,NA,3,NA,NA,NA,NA,3,NA,NA,2,NA,NA,NA,NA,3,NA,NA,NA,NA,2,3,NA,NA,NA,NA,3,NA,NA,NA,NA,NA,NA,NA,3,NA,NA,NA,2,NA,NA,NA,NA,NA,3,3,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,2,4,3,NA,4,NA,NA,2,NA,3,NA,2,NA,2,NA,NA,NA,NA,NA,3,NA,NA,2,3,NA,NA,NA,NA,2,NA,NA,NA,NA,NA,1,NA,NA,2,NA,NA,NA,NA,NA,NA,NA,3,NA,NA,4,2,NA,4,NA,NA,NA,NA,2,NA,5,NA,2,NA,3,NA,NA,NA,NA,NA,NA,NA,2,NA,3,NA,NA,3,NA,NA,5,NA,3,NA,NA,NA,NA,NA,NA,2,NA,3,2,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,3,4,2,2,NA,NA,1,NA,NA,3,NA,NA,NA,NA,NA,NA,NA,3,NA,NA,NA,3,NA,1,NA,2,2,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,2,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,3,NA,NA,NA,NA,NA,3,NA,NA,1,NA,NA,NA,NA,NA,NA,NA,3,3,NA,NA,3,NA,NA,5,NA,NA,NA,3,NA,NA,NA,NA,4,NA,NA,NA,NA,NA,NA,NA,NA,3,NA,NA,3,NA,2,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,NA,2,3,3,4,NA,3,NA,3,5,4,4,4,NA,5,3,3,NA,5,4,3,2,3,NA,NA,3,NA,3,3,5,5,3,NA,5,NA,3,3,NA,4,3,2,4,NA,NA,3,NA,3,NA,3,4,4,3,4,3,3,3,4,2,NA,3,3,3,3,5,5,4,NA,3,NA,2,NA,NA,NA,3,NA,3,3,NA,NA,2,3,NA,NA,3,NA,4,NA,NA,3,NA,4,NA,4,NA,NA,NA,NA,2,NA,NA,NA,NA,2,2,NA,NA,4,2,3,NA,2]
    print(len(u1181_ratings))
    existing_ratings = [pd.NA for _ in range(len(u1181_ratings))]
    for index, (key, rating) in enumerate(user_ratings.items()):
        if index in user_ratings:
            # Update the existing ratings list at the corresponding index based on the movie ID
            print(key)
            existing_ratings[index] = rating
    print(existing_ratings)
    return(myIBCF(existing_ratings))


def main():
    st.set_page_config(layout="wide")
    st.title("Movie Recommendation App")
    user_ratings = {}
    popular_movies = pd.read_csv("popular_movies.csv")
    movies_displayed = popular_movies.head(100)
    

    # Create columns for the grid layout
    num_columns = 5  # Number of movies per row
    columns = st.columns(num_columns)

    # Display movies in grid format with wider layout
    for index, row in movies_displayed.iterrows():
        col = columns[index % num_columns]  # Get the column for this movie
        
        with col:
            # Wrap the movie content inside a container
            with st.container():
                # Display the movie poster with title as caption
                image_url = f"{image_url_base}{row['MovieID']}.jpg"
                st.image(image_url, width=150, caption=f"'{row['Title']}'")
                
                # Display star ratings
                ratings = star_ratings(f"Rate '{row['Title']}'", numStars=5, key=row['Title'])

                # Show the rating text
                st.markdown("You rated %s stars" % int(ratings if ratings else 0))
                
                # Append the rating to the user ratings list
                user_ratings[row['MovieID']] = ratings

                print(user_ratings)


    if st.button("Get Recommendations"):
        recommendations = recommend_movies(user_ratings)
        st.write("Movies you may like:")
        for index, row in recommendations.iterrows():
            st.write(f"- {row['Title']}")


main()

