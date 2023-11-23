import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Read the CSV file into a Pandas DataFrame
csv_file_path = "assets/init_data.csv"
df = pd.read_csv(csv_file_path)

# Assuming the text data is in the 5th column, change the column index accordingly
text_data = df.iloc[:, 4]

# Create a TfidfVectorizer
vectorizer = TfidfVectorizer()

# Fit and transform the text data
tfidf_matrix = vectorizer.fit_transform(text_data)

# Get the feature names (words) from the vectorizer
feature_names = vectorizer.get_feature_names_out()

# Create a DataFrame with TF-IDF values for each word
tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

# Display the TF-IDF DataFrame
print(tfidf_df)
