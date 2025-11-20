# create_train_test_csv.py
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os


# Load params.yaml
param_file_path = "C:/Users/DELL/Desktop/my_ml_project/notebook/params.yaml"
with open(param_file_path, 'r') as f:
    params = yaml.safe_load(f)

# 1️⃣ Load your dataset
input_path = "notebook/fever.csv"  # adjust if needed
df_fever = pd.read_csv(input_path)
print("✅ Dataset loaded successfully:", df_fever.shape)

# 2️⃣ Define column groups
numerical_cols = ['Temperature', 'Age', 'BMI', 'Humidity', 'AQI', 'Heart_Rate']
categorical_cols = [
    'Gender', 'Headache', 'Body_Ache', 'Fatigue', 'Chronic_Conditions',
    'Allergies', 'Smoking_History', 'Alcohol_Consumption',
    'Physical_Activity', 'Diet_Type', 'Blood_Pressure',
    'Previous_Medication', 'Recommended_Medication'
]
target = 'Fever_Severity'  # 👈 use your real target column name here

# 3️⃣ Encode target if it's still text
if df_fever[target].dtype == 'object':
    print(f"⚙️ Encoding target column '{target}' as numeric codes...")
    le = LabelEncoder()
    df_fever['Fever_Severity_code'] = le.fit_transform(df_fever[target])
    print("✅ Encoding done. Mapping:", dict(zip(le.classes_, le.transform(le.classes_))))
else:
    df_fever['Fever_Severity_code'] = df_fever[target]
    print("✅ Target already numeric, no encoding needed.")

# 4️⃣ Split features and target
X = df_fever[numerical_cols + categorical_cols]
y = df_fever['Fever_Severity_code']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=params['preprocess']['test_size'], random_state=params['preprocess']['random_state'], stratify=y
)

# 5️⃣ Recombine and save
train_df = X_train.copy()
train_df['Fever_Severity_code'] = y_train

test_df = X_test.copy()
test_df['Fever_Severity_code'] = y_test

os.makedirs("data", exist_ok=True)
train_path = "data/processed/train.csv"
test_path = "data/processed/test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)

print(f"✅ Saved training and testing CSVs successfully!")
print(f"   → {train_path} ({train_df.shape[0]} rows)")
print(f"   → {test_path} ({test_df.shape[0]} rows)")
