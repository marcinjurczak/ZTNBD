# Binary Classification with Sonar Dataset: Baseline
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

df = pd.read_csv("Datasets/prepared_dataset.csv")
df_cleaned = df.drop(["Song Name", "Artist", "Release Date", "Chord"], axis=1)
df_cleaned["Genre"] = LabelEncoder().fit_transform(df_cleaned["Genre"])
df_cleaned['Streams'] = df_cleaned["Streams"].replace(regex=r',', value='')
df_cleaned["Streams"].replace({",": ""})
df_cleaned["Streams"] = pd.to_numeric(df_cleaned['Streams'])

print(df_cleaned.head())

X, y = df_cleaned.iloc[:, 1:].values, df_cleaned.iloc[:, 0].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train_norm = MinMaxScaler().fit_transform(X_train)
X_test_norm = MinMaxScaler().fit_transform(X_test)
encoder = LabelEncoder()
encoder.fit(y)
encoded_Y = encoder.transform(y)

model = Sequential()
model.add(Dense(32, input_dim=13, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train_norm, y_train, validation_split=0.1, batch_size=32, epochs=100)

print("Evaluate on test data")
results = model.evaluate(X_test_norm, y_test, batch_size=16)
print("test loss, test acc:", results)

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()