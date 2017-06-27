# Import pandas
import numpy as np
import pandas as pd

# Read in white wine data
white = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv", sep=';')

# Read in red wine data
red = pd.read_csv("http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv", sep=';')


# Add `type` column to `red` with value 1
red['type'] = 1

# Add `type` column to `white` with value 0
white['type'] = 0

# Append `white` to `red`
wines = red.append(white, ignore_index=True)


# Import `train_test_split` from `sklearn.model_selection`
from sklearn.model_selection import train_test_split

# Specify the data
X=wines.ix[:,0:11]

# Specify the target labels and flatten the array
y=np.ravel(wines.type)

# Split the data up in train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Import `StandardScaler` from `sklearn.preprocessing`
from sklearn.preprocessing import StandardScaler

# Define the scaler
scaler = StandardScaler().fit(X_train)


# Scale the train set
X_train = scaler.transform(X_train)

# Scale the test set
X_test = scaler.transform(X_test)


# Import `Sequential` from `keras.models`
from keras.models import Sequential

# Import `Dense` from `keras.layers`
from keras.layers import Dense

# Initialize the constructor
model = Sequential()

# Add an input layer
model.add(Dense(12, activation='relu', input_shape=(11,)))

# Add one hidden layer
model.add(Dense(8, activation='relu'))

# Add an output layer
model.add(Dense(1, activation='sigmoid'))


# Model output shape
#model.output_shape
# Model summary
#model.summary()
# Model config
#model.get_config()
# List all weight tensors
#model.get_weights()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=1)


#Predict values
y_pred = model.predict(X_test)


#evaluate Model
score = model.evaluate(X_test, y_test,verbose=1)
print(score)

print("\n"+"y-test: ")
print(y_test[:10])
print("\n"+"y-pred: ")
print(y_pred[:10])

# Import the modules from `sklearn.metrics`
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, cohen_kappa_score

# Confusion matrix
#confusion_matrix(y_test, y_pred)

# Precision
#precision_score(y_test, y_pred)

# Recall
#recall_score(y_test, y_pred)

# F1 score
#f1_score(y_test, y_pred)

# Cohen's kappa
#cohen_kappa_score(y_test, y_pred)