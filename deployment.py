import pandas as pd
import numpy as np
import pickle
#loading datasets
books = pd.read_csv('Datasets\BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM', 'imageUrlL']
books.drop('imageUrlS', axis=1, inplace=True)
books.drop('imageUrlM', axis=1, inplace=True)
books.drop('imageUrlL', axis=1, inplace=True)

#Removing Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))book from the dataset as there are 2 such instances
harry = books[books['bookTitle'] =="Harry Potter and the Sorcerer's Stone (Harry Potter (Paperback))"].index
books.drop(harry[0],inplace=True)
books.drop(harry[1],inplace=True)

#loading Datasets
ratings = pd.read_csv('Datasets\BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']

#Let's exclude users less than 200 ratings
counts1 = ratings['userID'].value_counts()
ratings = ratings[ratings['userID'].isin(counts1[counts1 >= 200].index)]

combine_book_rating = pd.merge(ratings,books,on='ISBN')
combine_book_rating.drop(['yearOfPublication','publisher','bookAuthor'],axis=1,inplace=True)
book_ratingCount = (combine_book_rating.
     groupby(by = ['bookTitle'])['bookRating'].
     count().
     reset_index().
     rename(columns = {'bookRating': 'totalRatingCount'})
     [['bookTitle', 'totalRatingCount']]
    )

rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on = 'bookTitle', right_on = 'bookTitle', how = 'left')

#including books with min rating count 50
popularity_threshold = 50
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')

#2D pivot table matrix -> create a sparse matrix
from scipy.sparse import csr_matrix
rating_popular_book = rating_popular_book.drop_duplicates(['userID', 'bookTitle'])
ratings_with_totalratings_pivot = rating_popular_book.pivot(index = 'bookTitle', columns = 'userID', values = 'bookRating').fillna(0)
ratings_with_totalratings_matrix = csr_matrix(ratings_with_totalratings_pivot.values)

#Cosine similarity
from sklearn.neighbors import NearestNeighbors
model = NearestNeighbors(metric='cosine',algorithm='brute')
model.fit(ratings_with_totalratings_matrix)

filename='knn-collab-recommendation.pkl'
pickle.dump(model,open(filename,'wb'))

knn_model=pickle.load(open(filename,'rb'))

ratings_with_totalratings_pivot = ratings_with_totalratings_pivot.reset_index()

def recommendation_knn(title):
    recommended_books=[]
    #Getting the bookId for the given title
    id = ratings_with_totalratings_pivot.loc[ratings_with_totalratings_pivot['bookTitle'] == title].index.values.astype(int)[0]
     #Getting the distances and indices of 5 nearest neighbours
    distances, indices = model.kneighbors(ratings_with_totalratings_pivot.drop('bookTitle',axis=1).iloc[id,:].values.reshape(1, -1), n_neighbors = 6)
    for i in range(1, len(distances.flatten())):
        recommended_books.append(ratings_with_totalratings_pivot.bookTitle[indices.flatten()[i]])
    return recommended_books
