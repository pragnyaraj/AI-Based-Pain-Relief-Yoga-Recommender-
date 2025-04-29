import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Load the updated CSV
df = pd.read_csv(r"C:\Users\pragn\YogaPoseRecommender\user_input_log.csv")

# Clean data
df["Pain Area"] = df["Pain Area"].str.lower().str.strip()
df["Pose"] = df["Pose"].str.lower().str.strip()

# Encode target variable (Pose)
df["Pose_Label"] = df["Pose"].astype("category").cat.codes
pose_mapping = dict(enumerate(df["Pose"].astype("category").cat.categories))

# Features and labels
X = df[["Pain Intensity"]]  # only intensity for now
y = df["Pose_Label"]        # pose as label

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train classifier
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

# Accuracy
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Example prediction
test_input = [[7]]
predicted_label = model.predict(test_input)[0]
print("Recommended Pose for intensity 7:", pose_mapping[predicted_label])
