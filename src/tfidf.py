import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import csv

csv_file_path = "assets/test.csv"
df = pd.read_csv(csv_file_path, quoting=csv.QUOTE_NONNUMERIC)

text_data = df.iloc[:, 2]

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(text_data)

feature_names = vectorizer.get_feature_names_out()

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

print(tfidf_df)
