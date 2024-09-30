import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import joblib


history = []
chat_data = []

def load_data(csv_file):
    data = pd.read_csv(csv_file)
    return data

def load_models():
    count_vectorizer = joblib.load(r"C:\Users\konre\CAPSTONE-PROJECT\search engine\models\count_vectorizer.joblib")
    tfidf_transformer = joblib.load(r"C:\Users\konre\CAPSTONE-PROJECT\search engine\models\tfidf_transformer.joblib")
    tfidf_matrix = joblib.load(r"C:\Users\konre\CAPSTONE-PROJECT\search engine\models\tfidf_matrix.joblib")
    return count_vectorizer, tfidf_transformer, tfidf_matrix

def retrieve_similar_documents(query, count_vectorizer, tfidf_transformer, tfidf_matrix, data, top_n=5):
    query_vector = count_vectorizer.transform([query])
    query_tfidf = tfidf_transformer.transform(query_vector)
    similarity_scores = cosine_similarity(query_tfidf, tfidf_matrix)
    top_indices = similarity_scores.argsort()[0][::-1]
    retrieved_documents = [data['clean_file_content'][idx] for idx in top_indices[:top_n]]
    retrieved_subtitle_names = [data['name'][idx] for idx in top_indices[:top_n]]  
    retrieved_subtitle_nums = [data['num'][idx] for idx in top_indices[:top_n]] 
    
    return retrieved_documents, retrieved_subtitle_names, retrieved_subtitle_nums

def main():
    global history, chat_data
    
    
    st.title('🎬🔍 FilmFinder')
    st.subheader('🍿In the realm of movies,allow SeekSpot to guide you through a cinematic adventure,effortlessly finding the films you crave🎥')

    
    st.sidebar.title('🌟 Navigation')
    if st.sidebar.button('🏠 Home'):
        st.sidebar.text('Go to Home')
        
        history = []
        chat_data = []
    if st.sidebar.button('📜 History'):
        st.sidebar.text('View Search History')
        
        st.sidebar.write(history)
    if st.sidebar.button('💾 Export'):
        st.sidebar.text('Export Data')
        
        export_data(chat_data)
    if st.sidebar.button('⚙️ Settings'):
        st.sidebar.text('Change Settings')

    query = st.text_input('Enter your query:', '')
    if st.button('🔎 Search'):
        if query:
            history.append(query)
            retrieved_documents, retrieved_subtitle_names, retrieved_subtitle_nums = retrieve_similar_documents(query, count_vectorizer, tfidf_transformer, tfidf_matrix, data)
            st.subheader('📑 Top 5 documents similar to the query:')
            for i, (doc, subtitle_name, subtitle_num) in enumerate(zip(retrieved_documents, retrieved_subtitle_names, retrieved_subtitle_nums), 1):
                st.write(f"**Document {i}:**")
                st.write(f"Subtitle Name: {subtitle_name}")
                st.write(f"Subtitle Number: {subtitle_num}")
                st.write("Summary:", doc)
            chat_data.append((query, retrieved_documents, retrieved_subtitle_names, retrieved_subtitle_nums))

def export_data(data):
    pass 

if __name__ == '__main__':
    data = load_data(r"C:\Users\konre\CAPSTONE-PROJECT\search engine\data\eng_subtitles_database.csv")
    count_vectorizer, tfidf_transformer, tfidf_matrix = load_models()
    main()
