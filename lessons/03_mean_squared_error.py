"""Lesson 3: one number for "how bad is this line?" — mean squared error.

MSE = average of (residual ** 2). Squaring makes every error positive and punishes
big misses more than small ones. Training means finding the m and b with the lowest MSE.

Run: python lessons/03_mean_squared_error.py
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


def mse(errors):
    squared_errors = []
    for error in errors:
        squared_errors.append(error**2)
    return sum(squared_errors) / len(squared_errors)


if __name__ == "__main__":
    experiences = [1, 2, 3, 4]
    actual_salaries = [35000, 40000, 45000, 50000]
    for m, b in [(1000, 10000), (4000, 25000), (5000, 30000)]:
        errors = residual(actual_salaries, predict(experiences, m, b))
        print(f"m={m:>5}  b={b:>6}  MSE={mse(errors):>16,.1f}")
    print("The last line fits the data exactly, so its MSE is 0.")
