import streamlit as st
import pandas as pd
import numpy as np

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

# Streamlit app
def search_data(data, column, keyword):
    return data[data[column].str.contains(keyword, case=False)]

def main():
    st.title("Random Data Search")

    # Search functionality
    search_column = st.selectbox("Select column to search:", df.columns)
    keyword = st.text_input("Enter keyword to search:")

    if keyword:
        search_results = search_data(df, search_column, keyword)
        st.dataframe(search_results)
    else:
        st.write("Enter a keyword to search.")

if __name__ == "__main__":
    main()
