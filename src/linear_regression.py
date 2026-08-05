import pickle

class LinearRegression:

    def __init__(self, learning_rate=0.000001, epochs=100):
        self.m = 0
        self.b = 0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, experiences):
        predictions = []
        for experience in experiences:
            predictions.append(self.m * experience + self.b)
        return predictions

    def _residual(self, actual, predicted):
        errors = []
        for index, actual_salary in enumerate(actual):
            errors.append(actual_salary - predicted[index])
        return errors

    def _mse(self, errors):
        squared_errors = []
        for error in errors:
            squared_errors.append(error ** 2)
        total = sum(squared_errors)
        return total / len(squared_errors)

    def _calculate_gradient_m(self, experiences, errors):
        gradient_m = 0
        for i in range(len(experiences)):
            gradient_m += experiences[i] * errors[i]

        gradient_m = (-2 / len(experiences)) * gradient_m
        return gradient_m

    def _calculate_gradient_b(self, errors):
        gradient_b = 0
        for error in errors:
            gradient_b += error
        gradient_b = (-2 / len(errors)) * gradient_b
        return gradient_b

    def fit(self, experiences, actual_salaries):

        # Training loop
        for epoch in range(self.epochs):
            predictions = self.predict(experiences)
            errors = self._residual(actual_salaries, predictions)
            loss = self._mse(errors)
            gradient_m = self._calculate_gradient_m(experiences, errors)
            gradient_b = self._calculate_gradient_b(errors)
            self.m = self.m - self.learning_rate * gradient_m
            self.b = self.b - self.learning_rate * gradient_b
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1} Loss: {loss}")

        print("\nTraining Completed")
        print("Learned m:", self.m)
        print("Learned b:", self.b)

    def save(self, file_name):
        with open(file_name, "wb") as file:
            pickle.dump(self, file)
        print("Model saved successfully.")

    @classmethod
    def load(cls, file_name):
        with open(file_name, "rb") as file:
            model = pickle.load(file)
        return model