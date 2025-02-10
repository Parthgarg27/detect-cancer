import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Load data
data = pd.read_csv('cancer_LAB4.csv')
data = data.drop(['id', 'Unnamed: 32'], axis=1)

# Prepare data
X = data.drop('diagnosis', axis=1)
y = data['diagnosis'].map({'M': 1, 'B': 0})  # Convert to binary labels

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models
models = {
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=10000),
    'Naive Bayes': GaussianNB()
}

# Train models
model_scores = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    predictions = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, predictions)
    model_scores[name] = (model, accuracy)

# Streamlit UI
st.set_page_config(page_title="Cancer Diagnosis Predictor", layout="wide", page_icon="🌊")
st.title(":rainbow[🌊 Cancer Diagnosis Predictor]")

# Sidebar instructions
with st.sidebar:
    st.header(":blue[🔹 Instructions]")
    st.markdown(":one: Enter the values for each feature below.")
    st.markdown(":two: Choose the model to make the prediction.")
    st.markdown(":three: Click **Predict** to see the diagnosis result.")

    st.header(":globe_with_meridians: Select Model")
    selected_model_name = st.selectbox("Choose Classification Model", list(models.keys()))

    st.header(":dart: Model Accuracies")
    for model_name, (_, accuracy) in model_scores.items():
        st.write(f"**{model_name}:** {accuracy*100:.2f}%")

# Custom CSS for styling
st.markdown("""
<style>
    .stButton>button {
        background-color: #6200EA;
        color: white;
        padding: 12px 30px;
        border-radius: 20px;
        border: none;
        cursor: pointer;
        font-size: 18px;
        font-weight: bold;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.2);
    }
    .stButton>button:hover {
        background-color: #3700B3;
    }
    .stTextInput>div>input, .stNumberInput>div>input {
        border-radius: 10px;
        padding: 10px;
    }
    .stNumberInput label {
        color: #6200EA; /* Change this to any color you prefer */
        font-weight: bold;
        font-size: 14px;
    }
    .stForm {
        background-color: #f0f0f0;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)


st.header(":calendar: Input Patient Data 🧑‍🦽")

mean_features = [feature for feature in X.columns if 'mean' in feature]
se_features = [feature for feature in X.columns if 'se' in feature]
worst_features = [feature for feature in X.columns if 'worst' in feature]

input_data = {}

with st.form(key='input_form'):
    cols = st.columns(3)

    with cols[0]:
        st.subheader(":orange[Mean Features]")
        for feature in mean_features:
            default_value = float(X[feature].mean())
            input_data[feature] = st.number_input(f"{feature}", value=default_value, step=0.01, format="%0.2f")

    with cols[1]:
        st.subheader(":orange[Standard Error Features]")
        for feature in se_features:
            default_value = float(X[feature].mean())
            input_data[feature] = st.number_input(f"{feature}", value=default_value, step=0.01, format="%0.2f")

    with cols[2]:
        st.subheader(":orange[Worst Features]")
        for feature in worst_features:
            default_value = float(X[feature].mean())
            input_data[feature] = st.number_input(f"{feature}", value=default_value, step=0.01, format="%0.2f")

    submit_button = st.form_submit_button(label='Predict Diagnosis')

# Prediction
if submit_button:
    selected_model = model_scores[selected_model_name][0]
    input_df = pd.DataFrame([input_data])
    input_scaled = scaler.transform(input_df)
    prediction = selected_model.predict(input_scaled)[0]
    diagnosis = 'Malignant (M)' if prediction == 1 else 'Benign (B)'

    st.markdown(f"""
    <div style='background-color: #f9f9f9; padding: 20px; border-radius: 10px; text-align: center;'>
        <h2 style='color: #333;'>Diagnosis Result</h2>
        <h1 style='color: {'#FF4B4B' if prediction == 1 else '#4CAF50'};'>{diagnosis}</h1>
        <p style='color: #555;'>Model Used: <strong>{selected_model_name}</strong> (Accuracy: {model_scores[selected_model_name][1]*100:.2f}%)</p>
    </div>
    """, unsafe_allow_html=True)
