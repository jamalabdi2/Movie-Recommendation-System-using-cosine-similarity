# Movie Recommendation System using cosine similarity

This project aims to create a movie recommendation system using the content-based approach. Given a user's favorite movie, the system recommends similar movies based on the movie's attributes.

# Table of Contents

1. Technologies
2. Dataset
3. Feature Selection
4. Recommendation System
5. Technologies

# This project uses the following libraries:

 1. Pandas
 2. Python
 3. difflib
 4. Scikit-learn

# Dataset
The dataset used in this project is the MovieLens Latest Full Dataset. It contains over 27,000 movies, 46,000 tags, and 26 million ratings.

# Feature Selection
The system uses the following attributes to recommend similar movies:

Genres
Keywords
Tagline
Cast
Director
The missing values in the dataset are replaced with empty strings.

# Recommendation System

The recommendation system works in the following way:

1. Combine the selected attributes of each movie into a single string.
2. Convert the strings to a sparse matrix of TF-IDF vectors.
3. Compute the cosine similarity matrix of the vectors.
4. Ask the user for their favorite movie and preprocess the input.
5. Find the closest match for the input movie name in the dataset.
6. Display the top 20 most similar movies and their similarity scores.
7. To run the code, open it in an environment that supports Python and the required libraries. Make sure to have the movies.csv file in the same directory as the code file. The user will be prompted to enter their favorite movie name to receive movie recommendations.

