# Make a prediction with coefficients
def predict(row, coefficients):
    yhat = coefficients[0]
    for i in range(len(row)-1):
        print(i)
        yhat += coefficients[i + 1] * row[i]
    return yhat
 
dataset = [[1, 1, 3], [2, 3, 4], [4, 3, 5], [3, 2, 1], [5, 5, 8]]
coef = [0.4, 0.8]
for row in dataset:
    yhat = predict(row, coef)
    print("Expected=%.3f, Predicted=%.3f" % (row[-1], yhat))