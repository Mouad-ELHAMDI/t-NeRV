"""Test t-SNE implementation on sklearn digits dataset."""
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from torchdr import TNERV as TorchdrTNERV

# Load digits dataset
digits = load_digits()
X = torch.tensor(digits.data, dtype=torch.float32)
y = digits.target

# Apply t-NeRV
print("Fitting t-NeRV...")
tnerv = TorchdrTNERV(
    perplexity=30.0,
    n_components=2,
    lr=200.0,
    max_iter=1000,
    early_exaggeration_coeff=12.0,
    early_exaggeration_iter=250,
    device="cuda",
    degrees_of_freedom=1,
    lambda_param=1
)
embedding = tnerv.fit_transform(X)

embedding = embedding.detach().numpy()
#print(type(embedding))

# Plot results
plt.figure(figsize=(10, 8))
scatter = plt.scatter(
    embedding[:, 0],
    embedding[:, 1],
    c=y,
    cmap='tab10',
    alpha=0.7,
    s=20
)
plt.colorbar(scatter, label='Digit')
plt.title('t-NeRV visualization of digits dataset')
plt.xlabel('t-NeRV dimension 1')
plt.ylabel('t-NeRV dimension 2')
plt.tight_layout()
plt.show()