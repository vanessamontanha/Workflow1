import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from joblib import dump

workspace = os.getenv('GITHUB_WORKSPACE')
path = os.path.join(workspace, 'ModelCleaning', 'cleaned_data.csv')

df = pd.read_csv(path)

X = df['age'].astype(float).values.reshape(-1, 1)
y = df['salary'].astype(float)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

dump(model, 'AgeSalaryModel.pkl')
