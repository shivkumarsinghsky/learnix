"""Lesson 2: residuals measure how wrong each prediction is.

residual = actual - predicted   (positive: we under-predicted)

Run: python lessons/02_residuals.py
"""


def predict(experiences, m, b):
    predictions = []
    for experience in experiences:
        predictions.append(m * experience + b)
    return predictions


def residual(actual, predicted):
    errors = []
    for index, actual_salary in enumerate(actual):
        errors.append(actual_salary - predicted[index])
    return errors


if __name__ == "__main__":
    m = 1000
    b = 10000
    experiences = [1, 2, 3, 4]
    actual_salaries = [35000, 40000, 45000, 50000]
    predicted_salary = predict(experiences, m, b)
    print("Prediction :", predicted_salary)
    print("Residual   :", residual(actual_salaries, predicted_salary))
