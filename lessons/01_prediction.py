"""Lesson 1: a prediction is just a line.

Salary is modelled as  salary = m * experience + b
  m = how much salary grows per year of experience (slope)
  b = salary with zero experience (intercept)

Run: python lessons/01_prediction.py
"""


def predict_salary(experiences, m, b):
    predictions = []
    for experience in experiences:
        predictions.append(m * experience + b)
    return predictions


if __name__ == "__main__":
    m = 5000
    b = 30000
    experiences = [2, 5, 8, 10, 15]
    print("Experience :", experiences)
    print("Prediction :", predict_salary(experiences, m, b))
