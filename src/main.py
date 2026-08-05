from linear_regression import LinearRegression

experiences = [1, 2, 3, 4]
actual_salaries = [35000, 40000, 45000, 50000]

model = LinearRegression()
model.fit(experiences, actual_salaries)
model.save("salary_model.pkl")

# Simulate tomorrow by creating a new variable
loaded_model = LinearRegression.load("salary_model.pkl")

print(loaded_model.predict([5]))