import pandas as pd


metadata = pd.read_csv('movies_metadata.csv', low_memory=False)

print(metadata['overview'].head())# head displays the first 5 rows, overview contains the plot of the movies

#this problem is a natural language processing problem so we need to use a feature
#that finds similarity and dissimilarity among them

#we will need to compute the word vectors of each document

from sklearn.feature_extraction.text import TfidfVectorizer

tfid = TfidfVectorizer(stop_words='english')
#this will Remove all english stop words

metadata['overview'] = metadata['overview'].fillna('') #replace Nan with empty string

#now constructing the required tf idf matric by transforming the data

tfid_matrix = tfid.fit_transform(metadata['overview'])

print(tfid_matrix.shape)
#(45466, 75827) these numbers mean that we observe 75827 different vocabuluries in our dataset of 45466 movies
print(tfid.get_feature_names_out()[5000:5010]) #this is a slice to get feature names for indices between this sequence

#with this matrice we can get the similarity score
from sklearn.metrics.pairwise import cosine_similarity

# Get the similarity of one movie (say index 0) to all others
cosine_sim = cosine_similarity(tfid_matrix[0], tfid_matrix).flatten()

# Get top 10 similar movie indices (excluding self)
indices = pd.Series(metadata.index, index=metadata['title']).drop_duplicates()

print(indices[:10])
#function take in movie title and output similiar movie
def get_recommendations(title, tfid_matrix=tfid_matrix, metadata=metadata):
    # Get the index of the movie that matches the title
    idx = indices[title]

    # Compute cosine similarity of that movie with all others
    sim_scores = cosine_similarity(tfid_matrix[idx], tfid_matrix).flatten()

    # Get the scores of the 10 most similar movies
    sim_scores = list(enumerate(sim_scores))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:11]  # Exclude the movie itself

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the top 10 similar movie titles
    return metadata['title'].iloc[movie_indices]

print(get_recommendations('The Godfather'))