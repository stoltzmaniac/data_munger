from single_linear_regression import fit_single_linear_regression, predict_single_linear_regression


x = [1, 2, 3]
y = [4, 5, 6]

model = fit_single_linear_regression(x, y)

x_predict = [5, 7, 10]
predictions = predict_single_linear_regression(x_predict, model)

print("#######")
print("Model")
print(model)
print("#######")
print(f"Predictions for {x_predict}")
print(predictions)
print("#######")