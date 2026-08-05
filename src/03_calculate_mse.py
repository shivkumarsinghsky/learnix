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
        squared_errors.append(error ** 2)
    total = sum(squared_errors)
    return total / len(squared_errors)


m = 1000
b = 10000
experiences = [1, 2, 3, 4]
actual_salaries = [35000, 40000, 45000, 50000]
predictions = predict(experiences, m, b)
errors = residual(actual_salaries, predictions)
loss = mse(errors)
print("Predictions :", predictions)
print("Residuals   :", errors)
print("MSE         :", loss)