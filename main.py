import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# 1. Generate Synthetic Data
def generate_synthetic_data(num_samples=1000):
    np.random.seed(42)
    heart_rate = np.random.randint(60, 120, num_samples)  # bpm
    calories_burned = np.random.randint(1500, 4000, num_samples)  # daily
    systolic_bp = np.random.randint(100, 180, num_samples)  # mmHg
    diastolic_bp = np.random.randint(60, 120, num_samples)  # mmHg
    labels = np.random.choice(['Low', 'Medium', 'High'], num_samples)  # Risk categories
    return pd.DataFrame({
        'Heart Rate': heart_rate,
        'Calories Burned': calories_burned,
        'Systolic BP': systolic_bp,
        'Diastolic BP': diastolic_bp,
        'Risk': labels
    })

# 2. Train Machine Learning Model
def train_model(data):
    X = data[['Heart Rate', 'Calories Burned', 'Systolic BP', 'Diastolic BP']]
    y = data['Risk']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    return model

# 3. Predict Health Risk
def predict_risk(model, inputs):
    return model.predict([inputs])[0]

# 4. Visualize Data
def visualize_metrics(data):
    sns.set(style="whitegrid")
    plt.figure(figsize=(10, 6))
    sns.boxplot(data=data[['Heart Rate', 'Calories Burned', 'Systolic BP', 'Diastolic BP']])
    plt.title("Distribution of Health Metrics")
    plt.show()

    sns.pairplot(data, hue='Risk', palette='coolwarm', diag_kind='kde')
    plt.title("Pairwise Relationships of Metrics")
    plt.show()

# Main Application
if __name__ == "__main__":
    # Generate synthetic data
    synthetic_data = generate_synthetic_data()

    # Train model
    model = train_model(synthetic_data)

    # Visualize metrics
    visualize_metrics(synthetic_data)

    # Predict a sample risk
    sample_input = [80, 2500, 140, 85]  # Heart Rate, Calories Burned, Systolic BP, Diastolic BP
    risk_prediction = predict_risk(model, sample_input)
    print(f"Predicted Risk for {sample_input}: {risk_prediction}")
