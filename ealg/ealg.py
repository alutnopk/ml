import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

class LogisticRegressor:
	'''Logistic regression class implemented only using numpy and pandas'''
	def __init__(self, learning_rate=0.01, num_iterations=1000, batch_size=32):
		self.learning_rate = learning_rate
		self.num_iterations = num_iterations
		self.batch_size = batch_size
		self.weights = None
		self.bias = None
		self.losses = []

	def sigmoid(self, z):
		return 1 / (1 + np.exp(-z))

	def cross_entropy_loss(self, y, y_pred):
		return (-1.0 / self.batch_size) * np.sum(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))

	def fit(self, X, y):
		'''Train the model based on the given inputs'''
		m, n = X.shape
		self.weights = np.zeros(n).astype(np.float64)
		self.bias = 0.0

		num_batches = m // self.batch_size

		for epoch in range(self.num_iterations):
			total_loss = 0.0
			# data shuffle
			permutation = np.random.permutation(m)
			X_shuffled = X[permutation]
			y_shuffled = y[permutation]

			for i in range(0, m, self.batch_size):
				# get mini-batch
				X_batch = X_shuffled[i:i+self.batch_size].astype(np.float64)
				y_batch = y_shuffled[i:i+self.batch_size].astype(np.float64)
				# forward prop
				z = np.dot(X_batch, self.weights) + self.bias
				y_pred = self.sigmoid(z)
				# cost calculation
				batch_loss = self.cross_entropy_loss(y_batch, y_pred)
				total_loss += batch_loss
				# back prop
				dw = (1/self.batch_size) * np.dot(X_batch.T, (y_pred - y_batch))
				db = (1/self.batch_size) * np.sum(y_pred - y_batch)
				# parameter update
				self.weights -= self.learning_rate * dw
				self.bias -= self.learning_rate * db

			# Compute average loss for epoch
			avg_loss = total_loss / num_batches
			self.losses.append(avg_loss)
			# print(f"Epoch {epoch+1}/{self.num_iterations}, Loss: {avg_loss}")

	def predict(self, X):
		'''Predict classification for a test dataset'''
		z = np.dot(X, self.weights) + self.bias
		y_pred = self.sigmoid(z)
		return np.round(y_pred)

	def plot_loss(self):
		'''Graph the variation of average loss with epoch'''
		plt.plot(range(1, self.num_iterations + 1), self.losses)
		plt.xlabel('Epoch')
		plt.ylabel('Loss')
		plt.title('Average Cross-Entropy Loss vs Epoch')
		plt.grid(True)
		plt.show()


# helper functions
def encode_cat_labels(X):
	'''Encode categorical features: one-hot encoding if non-binary, else binary encoding'''
	for col in X.columns:
		if X[col].dtype != 'object':
			continue
		unique_count_col = len(X[col].unique())
		if unique_count_col > 2:
			one_hot_cols = pd.get_dummies(X[col]).astype(int)
			X = pd.concat([X.drop(col, axis=1), one_hot_cols], axis=1)
		else:
			# X[col] = X[col].replace(X[col].unique(), [_ for _ in range(unique_count_col)])
			binary_mapping = {value: idx for idx, value in enumerate(X[col].unique())}
			X[col] = X[col].map(binary_mapping)
	return X
def z_score_normalize(X):
	'''Normalize all features using z-score'''
	scaler = StandardScaler()
	return scaler.fit_transform(X)

if __name__ == '__main__':
	df = pd.read_csv('dataset/hr-employee-attrition.csv')
	
	X = df.drop(columns=['Attrition']) # features
	y = (df['Attrition']=='Yes').astype(int).values # target variable

	X = encode_cat_labels(X).values
	X = z_score_normalize(X)
	
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

	ealg = LogisticRegressor()
	ealg.fit(X_train, y_train)

	y_pred = ealg.predict(X_test)

	precision = precision_score(y_pred, y_test)
	recall = recall_score(y_pred, y_test)
	f1 = f1_score(y_pred, y_test)
	accuracy = accuracy_score(y_pred, y_test)

	print("Precision:", precision)
	print("Recall:", recall)
	print("F1-score:", f1)
	print("Accuracy:", accuracy)

	ealg.plot_loss()
