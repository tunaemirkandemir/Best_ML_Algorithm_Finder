import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score

# Load CSV file
file_path = 'your_dataset.csv'  # Replace with the path to your CSV file
df = pd.read_csv(file_path)

# Assuming y column is the last column
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Enumerate the CSV
df['index'] = range(1, len(df) + 1)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize models
models = {
    'Logistic Regression': LogisticRegression(),
    'SVM': SVC(),
    'Naive Bayes': GaussianNB(),
    'Decision Trees': DecisionTreeClassifier(),
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestClassifier(),
    'Boosting': GradientBoostingClassifier()
}

# Train, evaluate, and fine-tune models
results = {}
for model_name, model in models.items():
    param_grid = {}

    if model_name == 'Logistic Regression':
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    elif model_name == 'SVM':
        param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'kernel': ['linear', 'rbf'], 'gamma': [0.1, 0.01, 0.001]}
    elif model_name == 'Decision Trees':
        param_grid = {'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10], 'min_samples_leaf': [1, 2, 4]}
    elif model_name == 'Random Forest':
        param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5],
                      'min_samples_leaf': [1, 2]}
    elif model_name == 'Boosting':
        param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}

    if param_grid:
        grid_search = GridSearchCV(model, param_grid, cv=5)
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
    else:
        model.fit(X_train, y_train)
        best_model = model

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    results[model_name] = accuracy

# Find the top three algorithms
top_three = sorted(results.items(), key=lambda x: x[1], reverse=True)[:3]

print("Accuracy Results:")
for model_name, accuracy in results.items():
    print(f"{model_name}: {accuracy}")

print("\nTop Three Algorithms:")
for model_name, accuracy in top_three:
    print(f"{model_name}: {accuracy}")



