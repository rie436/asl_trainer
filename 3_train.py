import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

df = pd.read_csv("asl_data.csv", header=None)
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

print("Samples per letter:")
print(df[0].value_counts())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

print("Test accuracy:", model.score(X_test, y_test))
joblib.dump(model, "asl_model.pkl")
print("Saved asl_model.pkl")