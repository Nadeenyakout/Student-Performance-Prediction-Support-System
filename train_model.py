#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Model imports
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


# In[6]:


get_ipython().system('pip install ucimlrepo')


# In[7]:


import pandas as pd
from ucimlrepo import fetch_ucirepo 

# fetch dataset 
student_performance = fetch_ucirepo(id=320) 
  
# data (as pandas dataframes) 
X = student_performance.data.features 
y = student_performance.data.targets 
  
# metadata 
print(student_performance.metadata) 
  
# variable information 
print(student_performance.variables) 


# In[8]:


# assume X, y from ucimlrepo were combined into df earlier, otherwise:
df = pd.concat([X, y], axis=1)

# 1. Inspect
print("shape:", df.shape)
display(df.head())
print("\nData types:")
print(df.dtypes)
print("\nMissing values per column:")
print(df.isnull().sum())


# In[9]:


import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# --- Step 1: Copy df to work safely ---
data = df.copy()

# --- Step 2: Drop irrelevant features ---
# G1 and G2 are highly correlated with G3 (final grade) -> remove to avoid "cheating"
data = data.drop(columns=['G1', 'G2'])

# --- Step 3: Encode categorical columns ---
categorical_cols = data.select_dtypes(include='object').columns
print("Categorical columns:", categorical_cols.tolist())

# Use One-Hot Encoding (turns categories into binary columns)
data_encoded = pd.get_dummies(data, columns=categorical_cols, drop_first=True)

# --- Step 4: Inspect final dataset ---
print("Shape after encoding:", data_encoded.shape)
display(data_encoded.head())


# In[10]:


# List all feature names except the target 'G3'
feature_names = [col for col in data_encoded.columns if col != 'G3']

print("Feature names used by the model:")
print(feature_names)


# In[11]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Assuming X and y are already loaded from ucimlrepo
df = pd.concat([X, y], axis=1)  # combine features + target for easier handling

# ----- Feature Engineering -----

# Total school days assumption (usually 365 days, but let’s assume ~200 school days in a year)
total_days = 200  

# Create Attendance Ratio
df['Attendance_Ratio'] = 1 - (df['absences'] / total_days)

# Create Average Grade
df['Average_Grade'] = (df['G1'] + df['G2'] + df['G3']) / 3

# ----- Splitting -----

# Define features (exclude final grade G3 if you want to predict it)
features = df.drop(columns=['G3'])  
target = df['G3']

# Train/test split (80/20)
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

print("Training set shape:", X_train.shape)
print("Test set shape:", X_test.shape)

# Check first few rows
df.head()


# In[12]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

# Assuming your original df is loaded
data = df.copy()

# Select only numeric features + target
features = ['age', 'Medu', 'Fedu', 'absences', 'studytime', 'failures', 'Dalc', 'health']
X = data[features]
y = data['G3']  # target

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[13]:


# Step 5 (Fixed): Encode categorical + Train Models
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

# Identify categorical + numeric columns
categorical_cols = X_train.select_dtypes(include=['object']).columns
numeric_cols = X_train.select_dtypes(exclude=['object']).columns

# Preprocessing: OneHotEncode categorical, pass through numeric
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols),
        ('num', 'passthrough', numeric_cols)
    ]
)

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM": SVC(random_state=42)
}

results = {}

# Train & Evaluate each with preprocessing pipeline
for name, model in models.items():
    clf = Pipeline(steps=[('preprocessor', preprocessor),
                          ('classifier', model)])
    
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-score": f1
    }

results_df = pd.DataFrame(results).T
print("\n✅ Model Evaluation Results:")
print(results_df)


# In[14]:


# Make sure y is 1D
y_series = data_encoded['G3']  # this is a Series

# Now create classes
y_class = pd.cut(y_series, bins=[-1, 9, 14, 20], labels=[0,1,2])


# In[15]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import joblib

# Assuming your original df is loaded
data = df.copy()

# Select only numeric features + target
features = ['age', 'Medu', 'Fedu', 'absences', 'studytime', 'failures', 'Dalc', 'health']
X = data[features]
y = data['G3']  # target

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(64,32), activation='relu', max_iter=300, random_state=42)
mlp.fit(X_train_scaled, y_train)

# Evaluate
y_pred = mlp.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"MLPClassifier Test Accuracy: {accuracy:.4f}")

# Save model and scaler for Flask
joblib.dump(mlp, "student_grade_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("Model and scaler saved!")


# In[16]:


pip install textblob


# In[17]:


from textblob import download_corpora

# Download required corpora
download_corpora.download_all()


# In[18]:


import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Simulate student feedback dataset ---
data_nlp = pd.DataFrame({
    'Feedback': [
        "I really enjoyed the course, learned a lot!",
        "The lectures were boring and confusing.",
        "Great teacher, but too much homework.",
        "I didn't understand most of the topics.",
        "Excellent explanations, very clear.",
        "Terrible experience, would not recommend.",
        "It was okay, not very engaging.",
        "Loved the activities and group work.",
        "The course was stressful and difficult.",
        "Fantastic! I feel confident in my skills now."
    ],
    'Grade': [18, 9, 14, 10, 19, 7, 12, 20, 8, 20]  # Simulated grades
})

# --- 2. Compute sentiment polarity ---
def get_sentiment(text):
    return TextBlob(text).sentiment.polarity  # -1 negative, 0 neutral, +1 positive

data_nlp['Sentiment'] = data_nlp['Feedback'].apply(get_sentiment)

# --- 3. Optional: convert to classes for analysis ---
# Negative (<0), Neutral (0-0.3), Positive (>0.3)
data_nlp['Sentiment_Class'] = pd.cut(
    data_nlp['Sentiment'],
    bins=[-1, 0, 0.3, 1],
    labels=['Negative', 'Neutral', 'Positive']
)

# --- 4. Analyze correlation with grades ---
sns.boxplot(x='Sentiment_Class', y='Grade', data=data_nlp)
plt.title("Student Grades vs Feedback Sentiment")
plt.show()

# Optional: correlation coefficient
corr = data_nlp[['Sentiment', 'Grade']].corr().iloc[0,1]
print(f"Correlation between sentiment polarity and grade: {corr:.2f}")

# Inspect data
print(data_nlp)


# In[19]:


import joblib

# Suppose mlp is your trained model
joblib.dump(mlp, "student_grade_model.pkl")
print("Model saved successfully!")


# In[20]:


joblib.dump(scaler, "scaler.pkl")


# In[21]:


pip install flask


# In[22]:


# ------------------------------
# Step 8: Deploying ML Model
# ------------------------------

import joblib
import requests
import pandas as pd

# --- 1. Save trained model & scaler ---
joblib.dump(mlp, "student_grade_model.pkl")
joblib.dump(scaler, "scaler.pkl")
print("✅ Model and scaler saved successfully!")

# --- 2. Flask App ---
# The Flask app is saved in a separate file called app.py.
# It loads the saved model & scaler, takes JSON input, and outputs:
# - Predicted Grade
# - Risk Category (High/Medium/Low)

# Example URL if running locally:
url = "http://127.0.0.1:5000/predict"

# --- 3. Example input data ---
example_student = {
    "age": 18,
    "Medu": 4,
    "Fedu": 4,
    "traveltime": 1,
    "studytime": 2,
    "failures": 0,
    "famrel": 4,
    "freetime": 3,
    "goout": 3,
    "Dalc": 1,
    "Walc": 1,
    "health": 5,
    "absences": 3,
    "school_MS": 1,
    "sex_M": 1,
    "address_U": 0,
    "famsize_LE3": 1,
    "Pstatus_T": 0,
    "Mjob_health": 0,
    "Mjob_other": 0,
    "Mjob_services": 1,
    "Mjob_teacher": 0,
    "Fjob_health": 0,
    "Fjob_other": 0,
    "Fjob_services": 0,
    "Fjob_teacher": 1,
    "reason_home": 0,
    "reason_other": 0,
    "reason_reputation": 1,
    "guardian_mother": 1,
    "guardian_other": 0,
    "schoolsup_yes": 1,
    "famsup_yes": 0,
    "paid_yes": 0,
    "activities_yes": 1,
    "nursery_yes": 1,
    "higher_yes": 1,
    "internet_yes": 1,
    "romantic_yes": 0
}

# --- 4. Send POST request to Flask API ---
response = requests.post(url, json=example_student)

# --- 5. Display API output ---
print("✅ Example prediction from Flask API:")
print(response.json())

# Optionally, test multiple students and store results
students = [example_student]  # you can add more dicts for other students
results = [requests.post(url, json=s).json() for s in students]
df_results = pd.DataFrame(results)
print("\n✅ Predictions for multiple students:")
display(df_results)


# In[ ]:


# Step 9: Ethics in AI

import pandas as pd

# --- 1. Check anonymization ---
# Display first few columns to ensure no personal identifiers
print("Columns in dataset:", df.columns.tolist())

# In this dataset, there are no names or IDs, only anonymous features
# ✅ Anonymization confirmed


# ### Bias Considerations
# 
# **Sensitive features** that might introduce bias:
# - Gender: `sex_M`
# - Socioeconomic status: `Medu`, `Fedu`, `famrel`, `famsize_LE3`, `Pstatus_T`
# - Access to resources: `internet_yes`, `schoolsup_yes`
# 
# Potential issues:
# - Model could favor one gender over another.
# - Students from wealthier families or with more parental education may receive better predicted grades.
# - Students without internet or school support could be unfairly predicted lower grades.
# 
# Mitigation strategies:
# - Monitor correlations between sensitive features and predicted grades.
# - Avoid including overly biased features, or reweight samples.
# - Consider fairness metrics like demographic parity in future work.
# 

# In[ ]:


# --- 2. Simple bias check ---
# Using model predictions from your trained MLP model

# Predict on the full dataset
features = data_encoded.drop(columns=['G3'])
X_scaled = scaler.transform(features)
predictions = mlp.predict(X_scaled)

# Convert to DataFrame for analysis
df_bias = data_encoded.copy()
df_bias['Predicted_Grade'] = predictions

# Example: check average predicted grade by gender
avg_by_gender = df_bias.groupby('sex_M')['Predicted_Grade'].mean()
print("Average Predicted Grade by Gender (0=Female,1=Male):")
print(avg_by_gender)

# Example: check average predicted grade by internet access
avg_by_internet = df_bias.groupby('internet_yes')['Predicted_Grade'].mean()
print("\nAverage Predicted Grade by Internet Access (0=No,1=Yes):")
print(avg_by_internet)


# **Observations & Recommendations:**
# - If large differences are observed between groups (e.g., males vs females, internet access vs no access), it indicates potential bias.
# - To mitigate bias:
#   - Consider removing or transforming sensitive features.
#   - Use fairness-aware algorithms.
#   - Monitor predictions continuously when deploying the model.
# 
