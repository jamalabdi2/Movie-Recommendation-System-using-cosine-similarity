

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import difflib
from sklearn.metrics.pairwise import cosine_similarity

from google.colab import files
files.upload()

movies = pd.read_csv('/content/movies.csv')
movies.head(3)

#shape of the datasets
movies.shape

#information about the dataset
movies.info()

#columns in the dataset
movies.columns

#data types the dataset
movies.dtypes

#statistics for numerical value
movies.describe()

#statistics for object value
movies.describe(include='object')

#checking for missing values
#statistics for numerical value
movies.isnull().sum()

columns_with_missing_values = movies.loc[:, movies.isnull().sum() > 0]
columns_with_missing_values.head()



#feature selection
selected_feature = ['genres','keywords','tagline','cast','director']
new_movies = movies[selected_feature] 
new_movies.head()

new_movies.isnull().sum()

#replacing missing values with empty string
new_movies = new_movies.fillna('')

new_movies.isnull().sum()

def combine_features(movies):
    """
    Combine multiple features of a movie into a single string.

    Args:
    movies (pandas.DataFrame): A dataframe with columns 'genres', 'keywords', 'tagline', 'cast', and 'director'.

    Returns:
    pandas.Series: A series of strings, where each string is the concatenation of the five features.

    """
    # Define a list of features to be combined
    features = ['genres', 'keywords', 'tagline', 'cast', 'director']

    # Initialize an empty string to store the combined features
    combined_features = ""

    # Iterate through each feature and concatenate its values with a space separator
    for feature in features:
        # Fill any missing values with an empty string and convert to string type
        combined_features += movies[feature].fillna('').astype(str) + ' '

    # Strip any leading/trailing whitespace and return the combined string as a pandas Series
    return combined_features.str.strip()

# Call the function on the new_movies dataframe
combined_features = combine_features(new_movies)
combined_features.head()



def get_feature_vector(data):
    """
    Convert a list of text data into a sparse matrix of TF-IDF vectors.

    Args:
    data (list): A list of strings to be vectorized.

    Returns:
    scipy.sparse.csr_matrix: A sparse matrix of shape (n_samples, n_features) representing the TF-IDF vectors of the input data.

    """

    # Define a TfidfVectorizer object to convert text to TF-IDF vectors
    vectorizer = TfidfVectorizer()

    # Fit the vectorizer on the input data and transform it into a sparse matrix
    feature_vector = vectorizer.fit_transform(data)

    return feature_vector

# Call the function on the combined features
feature_vector = get_feature_vector(combined_features)
print(feature_vector[10])

def get_cosine_similarity_matrix(feature_vector):
    """
    Compute the cosine similarity matrix of a feature vector.

    Args:
    feature_vector (scipy.sparse.csr_matrix): A sparse matrix of shape (n_samples, n_features) representing the feature vectors of a dataset.

    Returns:
    numpy.ndarray: A square matrix of shape (n_samples, n_samples) representing the pairwise cosine similarity scores between the samples.

    """

    # Compute the cosine similarity matrix using the feature vectors
    similarity_matrix = cosine_similarity(feature_vector)

    return similarity_matrix

# Call the function on the feature vector
similarity = get_cosine_similarity_matrix(feature_vector)
similarity[0]

title2 = [re.sub(r'[\s-]','',movies) for movies in movies['title']]
movies['title2'] = title2
movies['title2'] = movies['title2'].str.lower()

movies.head()

def find_similar_movies(movies, similarity):
    """
    This function takes in a Pandas DataFrame of movie titles and a cosine similarity matrix of the features of the movies
    and returns a list of similar movies based on the user's input movie name.

    Args:
    movies (Pandas DataFrame): DataFrame of movie titles and their features.
    similarity (numpy array): cosine similarity matrix of the features of the movies.

    Returns:
    Pandas DataFrame: DataFrame containing the top 20 most similar movies and their corresponding similarity score.
    """
    #getting input from the user
    movie_name = input('Enter Your favourite movie name: ')
    # preprocess user input and movie names
    movie_name = re.sub(r'\W+', '', movie_name.lower())

    #creating a list with all the movies names given in the dataset
    movie_list = movies['title2'].tolist()
    #finding the close match for the movie name given by the user
    get_close_match = difflib.get_close_matches(movie_name,movie_list)
    if len(get_close_match) == 0:

      # handle empty list
      print('No close matches found')
    else:
      close_match = get_close_match[0]

    if not get_close_match:
        print('Sorry, no close matches found for the given movie name.')
        return None
    else:
        print()
        # Print the close matches found
        print('\nClose matches:')
        for i, match in enumerate(get_close_match):
          print(f"{i+1}. {match}")
          print()
        selected_index_str = input("Enter the number corresponding to the movie you meant: ")
        try:
          selected_index = int(selected_index_str)
        except ValueError:
          print(f"Invalid input. Please enter a number between 1 and {len(get_close_match)}")

        close_match = get_close_match[0]
        accurate_close_match = get_close_match[selected_index-1] 
        print('\nFirst match is:', close_match)

        print('\nAccurate match is:',accurate_close_match)

        #find the index of the movie with title
        index_of_the_movie = movies[movies.title2 == accurate_close_match]['index'].values[0]
        print('\nindex of the movie:', index_of_the_movie)

        #getting a list of a similar movies
        similarity_score = list(enumerate(similarity[index_of_the_movie]))

        #sorting the movies based on the similarity score
        sorted_similar_movies = sorted(similarity_score, key= lambda x:x[1],reverse=True )

        #print the name of similar movies 
        print('Movies suggested for you:')
        titles = []
        for movie in sorted_similar_movies[:21]:
            index = movie[0]
            score = round(movie[1],3)
            title = movies[movies.index == index]['title'].values[0]
            titles.append({
                'Movie name':title,
                'Movie Index':index,
                'Similarity Score':score
            })
        movie_rank = pd.DataFrame(titles)
        return movie_rank

find_similar_movies(movies,similarity)