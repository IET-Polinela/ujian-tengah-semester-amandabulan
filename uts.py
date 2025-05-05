
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import os

df = pd.read_csv('/content/ujian-tengah-semester-amandabulan/healthcare-dataset-stroke-data.csv')

df = df.copy()
df['bmi'] = df['bmi'].fillna(df['bmi'].median())
df.drop('id', axis=1, inplace=True)

label_encoders = {}
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

X = df.drop('stroke', axis=1)
y = df['stroke']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42)

model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', scale_pos_weight=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

cm = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)

print("=== Confusion Matrix ===")
print(cm)
print("\n=== Classification Report ===")
print(cr)
print(f"=== Accuracy: {acc:.4f} ===")

feature_importance = model.feature_importances_
features = df.drop('stroke', axis=1).columns

plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance, y=features)
plt.title('Feature Importance - XGBoost')
plt.tight_layout()

output_path = '/content/ujian-tengah-semester-amandabulan/xgboost_feature_importance.png'
plt.savefig(output_path)
plt.close()

print("\n=== Visualisasi feature importance disimpan di:", output_path, "===")
