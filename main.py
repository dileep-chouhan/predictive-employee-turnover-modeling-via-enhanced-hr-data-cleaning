import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
# --- 1. Synthetic Data Generation ---
np.random.seed(42) # for reproducibility
n_employees = 500
data = {
    'Age': np.random.randint(20, 60, size=n_employees),
    'YearsExperience': np.random.randint(0, 30, size=n_employees),
    'SatisfactionScore': np.random.randint(1, 11, size=n_employees),  # 1-10 scale
    'WorkLifeBalance': np.random.randint(1, 11, size=n_employees), # 1-10 scale
    'PromotionLast5Years': np.random.randint(0, 2, size=n_employees), # 0 or 1
    'OverTime': np.random.choice(['Yes', 'No'], size=n_employees),
    'Left': np.random.randint(0, 2, size=n_employees) # 0: stayed, 1: left
}
df = pd.DataFrame(data)
# Introduce some missing data for cleaning practice
df.loc[np.random.choice(df.index, size=50), 'SatisfactionScore'] = np.nan
df.loc[np.random.choice(df.index, size=30), 'WorkLifeBalance'] = np.nan
# --- 2. Data Cleaning ---
# Handle missing values (simple imputation for demonstration)
df['SatisfactionScore'].fillna(df['SatisfactionScore'].mean(), inplace=True)
df['WorkLifeBalance'].fillna(df['WorkLifeBalance'].mean(), inplace=True)
# Convert categorical features to numerical using one-hot encoding
df = pd.get_dummies(df, columns=['OverTime'], drop_first=True) #drop_first to avoid multicollinearity
# --- 3. Data Preparation for Modeling ---
X = df.drop('Left', axis=1)
y = df['Left']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# --- 4. Model Training and Evaluation ---
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
# --- 5. Visualization ---
plt.figure(figsize=(8, 6))
sns.countplot(x='Left', data=df)
plt.title('Employee Turnover')
plt.xlabel('Left (0: Stayed, 1: Left)')
plt.ylabel('Number of Employees')
plt.savefig('employee_turnover.png')
print("Plot saved to employee_turnover.png")
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.savefig('correlation_matrix.png')
print("Plot saved to correlation_matrix.png")