import pandas as pd #using pandas for loading datasets for data manipulation and analysis, it is backed by numpy lib

metadata = pd.read_csv('movies_metadata.csv', low_memory=False) #not allowing memory optimization so that the file is not read in chunks but as whole

C = metadata['vote_average'].mean() #Calculates the average rating of a movie around 5.6/10
print(C)

#now we have to calculate the number of votes m in the 90th percentile

m = metadata['vote_count'].quantile(0.90)
print(m)
#160.0
#now we can filter out movies having equal or greatere then votes 160
#filter out all the movies with above condtion into a separate dataset

movies = metadata.copy().loc[metadata['vote_count'] >= m]
print(movies.shape) #shape returns number of row and column in a datasets
#4555 rows and 24 columns

#function to compute weighted rating of each movie
def weighted_rating(x , m = m, C=C):
    v = x['vote_count'] #v is vote counte
    R = x['vote_average']#R is vote average
    return (v/(v+m)*R) + (m/(m+v)*C) #formula from IMDB 

#defining a new feature score

movies['score'] = movies.apply(weighted_rating,axis=1) #applying to each row using axis 1 parameter
#Sort movies based on score calculated above
movies = movies.sort_values('score', ascending=False)

#Print the top 20 movies
movies[['title', 'vote_count', 'vote_average', 'score']].head(20).to_csv('top_20_bangers.csv',index=False) #head for 20 rows of sorted data
#index prevents pandas from adding unnecessary coloumns
