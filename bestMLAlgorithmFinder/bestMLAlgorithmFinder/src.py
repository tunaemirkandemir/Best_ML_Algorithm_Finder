import os

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LinearRegression

# 'your_dataset.extension' with your local file
file_path = 'your_dataset.extension'
file_extension = os.path.splitext(os.path.basename(file_path))[1]
if file_extension == '.json':

    json_file_path = file_path
    df = pd.read_json(json_file_path)

elif file_extension == '.csv':

    csv_file_path = file_path
    df = pd.read_csv(csv_file_path)

else:

    xlsx_file_name = file_path
    df = pd.read_excel(xlsx_file_name)

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
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        # Use cross_val_score for cross-validation with the best model
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='accuracy')

        # Store hyperparameter tuning data in the results dictionary
        results[model_name] = {
            'cv_scores': cv_scores,
            'best_model': best_model,
            'tuning_data': {
                'best_params': grid_search.best_params_,
                'all_cv_results': grid_search.cv_results_
            }
        }

    else:
        # Use cross_val_score for cross-validation with the model
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')

        # Store data for models without hyperparameter tuning
        results[model_name] = {
            'cv_scores': cv_scores,
            'best_model': model,
            'tuning_data': None  # No hyperparameter tuning data for models without tuning
        }

# Find the top algorithm based on mean cross-validation scores
top_algorithm = max(results.items(), key=lambda x: x[1]['cv_scores'].mean())

print("Cross-Validation Scores:")
for model_name, result in results.items():
    mean_cv_score = result['cv_scores'].mean()
    print(f"{model_name}: Mean CV Score - {mean_cv_score}")

print(f"\nTop Algorithm based on Mean CV Score: {top_algorithm[0]}")

# If the top algorithm underwent hyperparameter tuning, print tuning data
if top_algorithm[1]['tuning_data']:
    print(f"\nTuning Data for Top Algorithm ({top_algorithm[0]}):")
    print(f"Best Parameters: {top_algorithm[1]['tuning_data']['best_params']}")
    print(f"All CV Results: {top_algorithm[1]['tuning_data']['all_cv_results']}")