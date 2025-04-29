import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load the dataset
df = pd.read_csv(r"C:\Users\pragn\YogaPoseRecommender\pose_landmarks.csv")  # Use your actual CSV filename

# Ensure there are no missing values in the dataset
df = df.dropna()

# Split features and labels
X = df.drop(columns=["label"])  # Drop the 'label' column to separate features
y = df["label"]  # The target label column is 'label'

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the RandomForestClassifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate accuracy
accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy:.2f}")

# Save the trained model to a file
joblib.dump(model, "pose_classifier.pkl")
print("Model saved as pose_classifier.pkl")
