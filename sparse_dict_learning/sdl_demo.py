import numpy as np
from sklearn.decomposition import DictionaryLearning

# Generate synthetic data
np.random.seed(0)
Y = np.random.randn(100, 20)  # 100-dimensional data, 20 samples
print("Data:")
print(Y[:5, :])  # Print first 5 rows of data

# Perform sparse dictionary learning
dict_learner = DictionaryLearning(n_components=30, alpha=1, max_iter=500)
X = dict_learner.fit_transform(Y)
D = dict_learner.components_
# Reconstruct the data using the learned dictionary
Y_reconstructed = np.dot(X, D)
print("Sparse code:")
print(X)
print("Dictionary:")
print(D)
print("Reconstructed data:")
print(Y_reconstructed[:5, :])  # Print first 5 rows of reconstructed data

print("Dictionary shape:", D.shape)
print("Sparse code shape:", X.shape)