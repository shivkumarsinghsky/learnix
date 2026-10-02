"""Lesson 4: gradient descent finds m and b automatically.

Each epoch: predict -> residuals -> gradients -> step downhill.
  dMSE/dm = -(2/n) * sum(x * residual)
  dMSE/db = -(2/n) * sum(residual)
  m = m - learning_rate * dMSE/dm
  b = b - learning_rate * dMSE/db

The learning rate matters. Too small and training barely moves; too large and the loss
explodes. This lesson runs all three cases with the learnix implementation.

Run: python lessons/04_gradient_descent.py
"""

from learnix import LinearRegression, TrainingDivergedError

experiences = [1, 2, 3, 4]
actual_salaries = [35000, 40000, 45000, 50000]  # exactly salary = 5000 * x + 30000

print("1) learning_rate = 0.000001 (the first version of this lesson): far too small")
slow = LinearRegression(learning_rate=0.000001, epochs=100).fit(experiences, actual_salaries)
print(f"   after 100 epochs: m={slow.m:,.2f}  b={slow.b:,.2f}  (target m=5000, b=30000)\n")

print("2) learning_rate = 0.05: converges")
good = LinearRegression(learning_rate=0.05, epochs=5000).fit(
    experiences, actual_salaries, verbose=True
)
print(f"   learned m={good.m:,.2f}  b={good.b:,.2f}")
print(f"   prediction for 5 years: {good.predict([5])[0]:,.0f}\n")

print("3) learning_rate = 0.5: too large, the loss explodes")
try:
    LinearRegression(learning_rate=0.5, epochs=5000).fit(experiences, actual_salaries)
except TrainingDivergedError as error:
    print(f"   {error}")
