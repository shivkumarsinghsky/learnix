"""Lesson 5: train on a real-sized dataset, check the answer, save and reload the model.

The dataset (data/salary.csv, 50 rows) has experience from 1 to 23 years. With raw x values
that large, plain gradient descent needs a tiny learning rate; standardising x first lets it
converge quickly. We then compare against the exact least-squares solution.

Run: python lessons/05_train_save_load.py
"""

from pathlib import Path

from learnix import LinearRegression, fit_closed_form, load_csv_xy, mean_squared_error, r2_score

DATA = Path(__file__).resolve().parent.parent / "data" / "salary.csv"

experiences, salaries = load_csv_xy(DATA, "experience_years", "salary")

model = LinearRegression(learning_rate=0.1, epochs=2000, standardize=True, tolerance=1e-6)
model.fit(experiences, salaries)
exact_m, exact_b = fit_closed_form(experiences, salaries)
predictions = model.predict(experiences)

print(f"gradient descent : m={model.m:,.2f}  b={model.b:,.2f}  ({len(model.loss_history)} epochs)")
print(f"closed form      : m={exact_m:,.2f}  b={exact_b:,.2f}")
mse = mean_squared_error(salaries, predictions)
print(f"MSE={mse:,.0f}  R^2={r2_score(salaries, predictions):.4f}")

path = model.save("salary_model.json")
reloaded = LinearRegression.load(path)
print(f"saved to {path}; reloaded model predicts {reloaded.predict([5])[0]:,.0f} for 5 years")
