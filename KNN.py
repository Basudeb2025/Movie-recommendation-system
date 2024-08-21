#!/usr/bin/env python
# coding: utf-8

# In[265]:


import pandas as pd


# In[ ]:





# In[266]:


movies_name = pd.read_csv(r'C:\Users\Basudeb\OneDrive\Desktop\ml-latest-small\movies.csv')


# In[ ]:





# In[267]:


movies_ratings = pd.read_csv(r'C:\Users\Basudeb\OneDrive\Desktop\ml-latest-small\ratings.csv')


# In[ ]:





# In[ ]:





# In[165]:


print(movies_ratings.head())


# In[ ]:





# In[166]:


print(movies_name.head())


# In[ ]:





# In[268]:


merges_dataset = pd.merge(movies_ratings,movies_name,on = 'movieId')


# In[ ]:





# In[247]:


print(merges_dataset)


# In[ ]:





# In[ ]:





# In[36]:





# In[ ]:





# In[ ]:


#there can be many movieId in the row,so for delete this, use the mean of all rating and keep one


# In[269]:


delete_duplicate_data = merges_dataset.groupby(['movieId', 'title']).agg({'rating': 'mean'}).reset_index()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#row are movieId, columns are movie name, and values represent ratings for each movie


# In[270]:


final_Data = delete_duplicate_data.pivot(index = 'movieId',columns = 'title',values = 'rating')


# In[ ]:





# In[248]:


print(final_Data)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[152]:


print(final_Data.T.shape[0])


# In[ ]:





# In[150]:


print(final_Data.shape)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


#For missing data


# In[271]:


final_Data = final_Data.fillna(0)


# In[ ]:





# In[ ]:





# In[227]:


print(list(final_Data.columns).index('Toy Story (1995)'))


# In[ ]:





# In[ ]:





# In[ ]:


#KNN part


# In[272]:


from sklearn.neighbors import NearestNeighbors


# In[ ]:





# In[273]:


k = 7


# In[ ]:





# In[274]:


knn = NearestNeighbors(n_neighbors = 7,metric = 'cosine')


# In[ ]:





# In[275]:


final_Data_T = final_Data.T # for item based 


# In[ ]:





# In[276]:


knn.fit(final_Data_T)


# In[ ]:





# In[ ]:





# In[277]:


def similar_movie(movie_name, k=5):
    # Check if the movie is not present in columns
    if movie_name not in final_Data_T.index:
        return 'The movie is not present'
    
    # Get the index of the movie_name
    try:
        movie_index = list(final_Data_T.index).index(movie_name)
    except ValueError as e:
        return f'Error finding movie index: {e}'
    
    # Create the rating file for the movie
    try:
        ratings_file = final_Data_T.iloc[movie_index].values.reshape(1, -1)
    except IndexError as e:
        return f'IndexError: {e}'
    
    # Calculate distances and indices
    distances, indices = knn.kneighbors(ratings_file, n_neighbors=k+1)
    # converting 2D to 1D array
    same_indices = indices.flatten()[1:]  # Exclude the movie itself
    #taking pair for {distance,index}
    pair = []
    for i in range(len(indices)):
        pair.append((distances[i], indices[i]))
    #sort the array based on the less distance
    pair.sort(key=lambda x: x[0])
     #same_movies.this array is storing the movies list which are present on the first element of pair
    same_movies = []
    for ele in pair:
        same_movies.append(final_Data.columns[ele[1]])
    #for pandas library its giving us 
    return same_movies[0].tolist()
   


# In[ ]:





# In[278]:


movie_id = 'Waiting to Exhale (1995)'
result = similar_movie(movie_id)
print(f"Similar Movies: {result}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[157]:


print(final_Data.iloc[8871])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




