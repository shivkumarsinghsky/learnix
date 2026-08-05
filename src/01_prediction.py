def predictSalary(x, m, b):
    predictions = []
    for experience in experiences:
        predictions.append(m * experience + b)
    return predictions

m = 5000
b = 30000
experiences = [2, 5, 8, 10, 15]
prediction = predictSalary(experiences, m, b)
print(prediction)