import pandas as pd
import numpy as np
import pickle

filename='knn-collab-recommendation.pkl'
knn_model=pickle.load(open(filename,'rb'))
pickle.dump(knn_model,open(filename,'wb'))
df=pd.read_csv('CleanBookss.csv')
df.head()


def recommendation_knn(title):
    recommended_books=[]
    #Getting the bookId for the given title
    id = df.loc[df['BookTitle'] == title].index.values.astype(int)[0]
     #Getting the distances and indices of 5 nearest neighbours
    distances, indices = knn_model.kneighbors(df.drop('BookTitle',axis=1).iloc[id,:].values.reshape(1, -1), n_neighbors = 6)
    for i in range(1, len(distances.flatten())):
        recommended_books.append(df.BookTitle[indices.flatten()[i]])
    return recommended_books
