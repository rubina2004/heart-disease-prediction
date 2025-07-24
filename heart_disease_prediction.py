
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. Load the dataset
heart_data = pd.read_csv('heart.csv')  # Update path if needed

print("\nInitial Data Snapshot:")
print(heart_data.head())

# 2. Handle missing values — drop rows with missing values
heart_data = heart_data.dropna()

print("\nData after dropping missing values:")
print(heart_data.info())

# 3. Encode categorical columns to numeric
categorical_columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'thal']

le = LabelEncoder()
for col in categorical_columns:
    heart_data[col] = le.fit_transform(heart_data[col])

print("\nData after encoding categorical columns:")
print(heart_data.head())

# 4. Rename target column 'num' → 'target'
heart_data.rename(columns={'num': 'target'}, inplace=True)

# 5. Binarize the target column
heart_data['target'] = heart_data['target'].apply(lambda x: 1 if x > 0 else 0)

print("\nTarget distribution after binarizing:")
print(heart_data['target'].value_counts())

# 6. Drop irrelevant columns if needed
heart_data = heart_data.drop(columns=['id', 'dataset'], errors='ignore')

# 7. Split features and target
X = heart_data.drop(columns='target', axis=1)
Y = heart_data['target']

# 8. Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

print("\nDataset shapes - Full: {}, Train: {}, Test: {}".format(
    X.shape, x_train.shape, x_test.shape))

# 9. Train Logistic Regression Model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)

# 10. Evaluate model
x_train_prediction = model.predict(x_train)
training_data_accuracy = accuracy_score(y_train, x_train_prediction)

x_test_prediction = model.predict(x_test)
test_data_accuracy = accuracy_score(y_test, x_test_prediction)

print('\nAccuracy on Training data:', training_data_accuracy)
print('Accuracy on Test data:', test_data_accuracy)

# 11. Predict for custom input
input_data = (63, 1, 3, 145, 233, 1, 2, 150, 0, 2.3, 0, 0.0, 2, 2)
# Example: (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

prediction = model.predict(input_data_reshaped)

print("\nPrediction for input data:")
if prediction[0] == 0:
    print('The person does NOT have a heart disease.')
else:
    print('The person HAS heart disease.')
