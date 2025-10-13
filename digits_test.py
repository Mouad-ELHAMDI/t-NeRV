"""Test t-SNE implementation on sklearn digits dataset."""
import torch
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from torchdr import TSNE as TorchdrTSNE

# Load digits dataset
digits = load_digits()
X = torch.tensor(digits.data, dtype=torch.float32)
y = digits.target

# Apply t-SNE
print("Fitting t-SNE...")
tsne = TorchdrTSNE(
    perplexity=30.0,
    n_components=2,
    lr=200.0,
    max_iter=1000,
    early_exaggeration_coeff=12.0,
    early_exaggeration_iter=250,
    device="cuda"
)
embedding = tsne.fit_transform(X)

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
plt.title('t-SNE visualization of digits dataset')
plt.xlabel('t-SNE dimension 1')
plt.ylabel('t-SNE dimension 2')
plt.tight_layout()
plt.savefig('tsne_digits.png', dpi=150)
print("Plot saved as 'tsne_digits.png'")
plt.show()