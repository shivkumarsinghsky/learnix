import joblib
model = joblib.load("notebooks/salary_model.pkl")
predictions = model.predict(4)
print(predictions)