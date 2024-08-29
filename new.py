from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import pandas as pd

# Load the dataset
df = pd.read_csv('students_placement.csv')
x = df.drop(['placed'], axis=1)
y = df['placed']

# Create a pipeline with an imputer and the model
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # or 'median', 'most_frequent'
    ('classifier', RandomForestClassifier())
])

# Split the data
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=2)

# Train the model with imputation
pipeline.fit(x_train, y_train)

# Save the model
joblib.dump(pipeline, 'placement_model.pkl')