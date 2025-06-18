import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.under_sampling import RandomUnderSampler
import joblib

# Step 1: Load Dataset
df = pd.read_csv("data/Steel_industry_data.csv")

# Step 2: Preprocess Date Column
df['date'] = pd.to_datetime(df['date'], dayfirst=True)
df.set_index('date', inplace=True)
df.sort_index(inplace=True)

# Step 3: Feature-Target Split
X = df.drop(columns=['Load_Type'])
y = df['Load_Type']

# Step 4: Categorical Encoding (if needed)
cats = X.select_dtypes(include=['object', 'bool']).columns
if len(cats) > 0:
    le = LabelEncoder()
    for col in cats:
        X[col] = le.fit_transform(X[col])

# Step 5: Target Encoding
le_target = LabelEncoder()
y = le_target.fit_transform(y)

# Step 6: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 7: Handling Imbalanced Classes (Under Sampling)
undersampler = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

# Step 8: Build Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Step 9: Train Model
pipeline.fit(X_train_resampled, y_train_resampled)

# Step 10: Evaluate
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("\n=== Evaluation Metrics: Random Forest Classifier ===")
print(f"Accuracy:  {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall:    {recall:.2f}")
print(f"F1-Score:  {f1:.2f}")

# Step 11: Save Best Model
joblib.dump(pipeline, "model_rf.joblib")