import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pip install scikit-learn


# Generate a random DataFrame for demonstration
np.random.seed(42)
data = {
    'ID': np.arange(1, 101),
    'Name': [f'Patient_{i}' for i in range(1, 101)],
    'Condition': np.random.choice(['Cancer', 'Diabetes', 'Heart Disease'], 100),
    'Medication': np.random.choice(['Med_A', 'Med_B', 'Med_C'], 100),
    'Age': np.random.randint(20, 80, 100)
}
df = pd.DataFrame(data)

# Function to perform keyword search and calculate relevance ranking
def search_data(data, column, keyword):
    results = data[data[column].str.contains(keyword, case=False)]
    if results.empty:
        return results

    # TF-IDF calculation
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(results[column])
    query_tfidf = tfidf.transform([keyword])
    cosine_sim = cosine_similarity(query_tfidf, tfidf_matrix)
    results['Relevance'] = cosine_sim.flatten()

    # Sort results by relevance
    results = results.sort_values(by='Relevance', ascending=False)
    return results

def main():
    st.title("Enhanced Data Search")

    # Search functionality
    search_column = st.selectbox("Select column to search:", df.columns)
    keyword = st.text_input("Enter keyword to search:")

    if keyword:
        search_results = search_data(df, search_column, keyword)
        if not search_results.empty:
            st.subheader("Search Results")
            st.dataframe(search_results)

            # Relevance Ranking
            st.subheader("Relevance Ranking")
            st.write("Top 5 relevant results:")
            st.write(search_results[['ID', 'Name', 'Relevance']].head(5))

            # Basic Statistics
            st.subheader("Basic Statistics")
            st.write("Number of results:", len(search_results))
            st.write("Average age of results:", search_results['Age'].mean())

            # TF-IDF Calculation
            st.subheader("TF-IDF Scores")
            st.write("TF-IDF scores for the search query:")
            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform(search_results[search_column])
            query_tfidf = tfidf.transform([keyword])
            feature_names = tfidf.get_feature_names_out()
            query_tfidf_df = pd.DataFrame(query_tfidf.toarray(), columns=feature_names)
            st.dataframe(query_tfidf_df)

        else:
            st.write("No results found for the given keyword.")
    else:
        st.write("Enter a keyword to search.")

if __name__ == "__main__":
    main()
