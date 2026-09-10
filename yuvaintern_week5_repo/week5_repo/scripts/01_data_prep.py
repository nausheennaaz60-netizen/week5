import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)

digits = load_digits()
X = digits.images  # (1797, 8, 8), pixel values 0-16
y = digits.target  # (1797,), digits 0-9

print("=== DATASET OVERVIEW ===")
print(f"Total images: {X.shape[0]}")
print(f"Image shape: {X.shape[1]}x{X.shape[2]} pixels, grayscale, values 0-16")
print(f"Classes: {sorted(set(y))}")
print(f"Class distribution: {np.bincount(y)}")

# Normalize pixel values to [0, 1] and add channel dimension for CNN input
X_norm = X.astype('float32') / 16.0
X_norm = X_norm.reshape(-1, 8, 8, 1)

# Split: 70% train, 15% validation, 15% test (stratified to preserve class balance)
X_train, X_temp, y_train, y_temp = train_test_split(
    X_norm, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"\nTrain: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")

# Visualize sample digits
fig, axes = plt.subplots(2, 10, figsize=(14, 3.2))
for digit in range(10):
    idx = np.where(y == digit)[0][0]
    axes[0, digit].imshow(X[idx], cmap='gray_r')
    axes[0, digit].set_title(str(digit), fontsize=10)
    axes[0, digit].axis('off')
    idx2 = np.where(y == digit)[0][1]
    axes[1, digit].imshow(X[idx2], cmap='gray_r')
    axes[1, digit].axis('off')
plt.suptitle('Sample Digits from Each Class (8x8 pixels)', fontweight='bold')
plt.tight_layout()
plt.savefig('fig1_sample_digits.png', dpi=120)
plt.close()

np.savez('digits_split.npz', X_train=X_train, X_val=X_val, X_test=X_test,
         y_train=y_train, y_val=y_val, y_test=y_test)
print("\nSaved split data and sample visualization.")
