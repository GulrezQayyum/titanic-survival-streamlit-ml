import pandas as pd

# Load dataset
data = pd.read_csv('train.csv')

# See the first 5 rows
print(data.head())

# Get info about columns
print(data.info())

# Check for missing values
print(data.isnull().sum())

# Drop columns we won't use
data = data.drop(['Name', 'Ticket', 'Cabin'], axis=1)

# Fill missing Age with median
data['Age'].fillna(data['Age'].median(), inplace=True)

# Fill missing Embarked with mode
data['Embarked'].fillna(data['Embarked'].mode()[0], inplace=True)
# Map 'Sex' to numeric
data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})

# One-hot encode Embarked
data = pd.get_dummies(data, columns=['Embarked'], drop_first=True)
X = data.drop('Survived', axis=1)
y = data['Survived']
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Initialize model
model = LogisticRegression(max_iter=1000)

# Train model
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
