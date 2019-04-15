import numpy as np
import sys

def weight_regression(lambda_input, sigma2_input, X_train, y_train, dim):
	## wRR = (λI + X^TX)^−1X^Ty
	wRR = (np.linalg.inv(lambda_input * np.eye(dim) + (X_train.T).dot(X_train)).dot((X_train.T).dot(y_train)))
	return wRR
	
def update_posterior(lambda_input, sigma2_input, data, label, dim):
	## covariance(Σ) = (λI + σ^−2X^TX)^−1;
	## mean(µ) = (λσ^2I + X^TX)^−1X^Ty
	covariance = np.linalg.inv(lambda_input * np.eye(dim) + (1/sigma2_input) * (data.T).dot(data))
	mean = np.linalg.inv((lambda_input * sigma2_input * np.eye(dim)) + ((data.T).dot(data))).dot((data.T).dot(label))
	return covariance, mean

def active_learning(lambda_input, sigma2_input, X_train, y_train, X_test, dim):
	covariance, mean = update_posterior(lambda_input, sigma2_input, X_train, y_train, dim)
	wRR = mean

	active_rows = []
	indices = list(range(X_test.shape[0]))

	for i in range(0, 10):
		##σ2 = x0^TΣx0
		sigma = (X_test.dot(covariance)).dot(X_test.T)
		row = np.argmax(sigma.diagonal())
		data = X_test[row, :]
		label = data.dot(wRR)
		active_rows.append(indices[row])
		#Remove the appended row
		X_test = np.delete(X_test, row, axis = 0)
		indices.pop(row)

		covariance, mean = update_posterior(lambda_input, sigma2_input, data, label, dim)
		wRR = mean

	## final list of values to write in the file
	active_rows = [j+1 for j in active_rows]
	return active_rows

def main():
	lambda_input = int(sys.argv[1])
	sigma2_input = float(sys.argv[2])

	X_train = np.genfromtxt(sys.argv[3], delimiter = ",")
	y_train = np.genfromtxt(sys.argv[4])
	X_test = np.genfromtxt(sys.argv[5], delimiter = ",")

	dim = X_train.shape[1]

	wRR = weight_regression(lambda_input, sigma2_input, X_train, y_train, dim)
	active_rows = active_learning(lambda_input, sigma2_input, X_train, y_train, X_test.copy(), dim)

	## save output to file
	np.savetxt("wRR_" + str(lambda_input) + ".csv", wRR, delimiter="\n")
	np.savetxt("active_" + str(lambda_input) + "_" + str(int(sigma2_input)) + ".csv", active_rows, delimiter=",")

if __name__ == "__main__":
	main()

