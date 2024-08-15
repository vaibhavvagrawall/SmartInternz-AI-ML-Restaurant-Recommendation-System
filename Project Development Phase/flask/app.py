from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import string

app = Flask(__name__)

# ... (Your existing code for data preprocessing and recommendation functions)

# Function to clean and preprocess text data
def preprocess_text(text):
    text = text.lower().translate(str.maketrans('', '', string.punctuation))
    return text

# Function to get top words from text series
def get_top_words(text_series, top_n, ngram_range):
    vectorizer = CountVectorizer(ngram_range=ngram_range, stop_words='english')
    matrix = vectorizer.fit_transform(text_series)
    word_count = np.asarray(matrix.sum(axis=0)).ravel()
    words = np.array(vectorizer.get_feature_names_out())
    words_df = pd.DataFrame({'word': words, 'count': word_count})
    return words_df.sort_values(by='count', ascending=False).head(top_n)

# Load the dataset
zomato_df = pd.read_csv('zomato.csv')

# Data Cleaning and Preprocessing
zomato_df = zomato_df.drop(columns=['url', 'phone', 'dish_liked'])
zomato_df.dropna(how='any', inplace=True)
zomato_df.drop_duplicates(inplace=True)

zomato_df = zomato_df.rename(columns={'approx_cost(for two people)': 'cost',
                                      'listed_in(type)': 'type',
                                      'listed_in(city)': 'city'})

zomato_df['rate'] = pd.to_numeric(zomato_df['rate'].str.replace('/5', '').str.strip(), errors='coerce')
zomato_df['cost'] = zomato_df['cost'].str.replace(',', '.').astype(float)

zomato_df['reviews_list'] = zomato_df['reviews_list'].apply(preprocess_text)
zomato_df['cuisines'] = zomato_df['cuisines'].apply(preprocess_text)


# Group by 'name' column and calculate the mean of 'rate' column
grouped_restaurants = zomato_df.groupby('name', as_index=False)['rate'].mean().round(2)

# Combine all reviews and cuisines for each unique name
combined_restaurants = zomato_df.groupby('name', as_index=False).agg({'reviews_list': 'sum', 'cuisines': 'sum'})

# Merge the two dataframes on 'name' column
merged_restaurants = pd.merge(grouped_restaurants, combined_restaurants, on='name')

# Create a 'tags' column
merged_restaurants['tags'] = merged_restaurants['reviews_list'] + merged_restaurants['cuisines']

# Drop unnecessary columns
final_df = merged_restaurants.drop(columns=['reviews_list', 'cuisines'])

# Add 'cost' column to the new dataframe
final_df['cost'] = zomato_df['cost']

# Use CountVectorizer for text vectorization
vectorizer = CountVectorizer(max_features=5000, stop_words='english')
tags_vector = vectorizer.fit_transform(final_df['tags']).toarray()

# Calculate cosine similarity
similarity_matrix = cosine_similarity(tags_vector)

# Function to recommend restaurants
def recommend_similar_restaurants(target_restaurant):
    target_restaurant_lower = preprocess_text(target_restaurant)

    if target_restaurant_lower not in final_df['name'].str.lower().values:
        print(f"Restaurant '{target_restaurant}' data not found.")
        return []

    target_index = final_df[final_df['name'].str.lower() == target_restaurant_lower].index[0]
    distances = sorted(enumerate(similarity_matrix[target_index]), reverse=True, key=lambda x: x[1])

    recommended_restaurants = [final_df.iloc[i[0]]['name'] for i in distances[1:11]]
    return recommended_restaurants


# Save dataframe and similarity matrix to pickle files
final_df.to_pickle('final_restaurants.pkl')
np.save('similarity_matrix.npy', similarity_matrix)



# Flask routes
@app.route('/')
def index():
    return render_template('index.html')
@app.route('/recommend', methods=['POST'])
def recommend():
    target_restaurant = request.form['restaurant']
    recommended_restaurants = recommend_similar_restaurants(target_restaurant)

    if recommended_restaurants is not None:
        return render_template('recommendations.html', restaurant_name=target_restaurant, recommended_restaurants=recommended_restaurants)
    else:
        return render_template('recommendations.html', restaurant_name=target_restaurant, recommended_restaurants=[])

if __name__ == "__main__":
    app.run(debug=True)
