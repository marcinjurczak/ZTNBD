import pandas as pd
from locale import atof
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv("Datasets/prepared_dataset.csv")
df_cleaned = df.drop(["Song Name", "Artist", "Release Date", "Chord"], axis=1) #remove inrelevant columns
df_cleaned["Genre"] = LabelEncoder().fit_transform(df_cleaned["Genre"])
df_cleaned['Streams'] = df_cleaned["Streams"].replace(regex=r',', value='')
df_cleaned["Streams"].replace({",": ""})
df_cleaned["Streams"] = pd.to_numeric(df_cleaned['Streams'])

X, y = df_cleaned.iloc[:, 1:].values, df_cleaned.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
X_train_norm = MinMaxScaler().fit_transform(X_train)
X_test_norm = MinMaxScaler().fit_transform(X_test)


classifiers = {
	"DecisionTreeClassifier": DecisionTreeClassifier(),
	"RandomForestClassifer": RandomForestClassifier(),
	"MultinomialNB": MultinomialNB(),
	"LogisticRegression": LogisticRegression(penalty="l2")
}

for name, clf in classifiers.items():
	clf.fit(X_train_norm, y_train)
	print(f"Used classifier: {name}")
	print(f"Train: {clf.score(X_train_norm, y_train)} \n Test: {clf.score(X_test_norm, y_test)}")

