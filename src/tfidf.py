import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

csv_file_path = "assets/init_data.csv"
df = pd.read_csv(csv_file_path)

text_data = df.iloc[:, 4]

vectorizer = TfidfVectorizer()

tfidf_matrix = vectorizer.fit_transform(text_data)

feature_names = vectorizer.get_feature_names_out()

tfidf_df = pd.DataFrame(tfidf_matrix.toarray(), columns=feature_names)

print(tfidf_df)
