import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

sns.set_style('whitegrid')

data = np.load('digits_split.npz')
X_test, y_test = data['X_test'], data['y_test']

model = keras.models.load_model('final_cnn_model.keras')

y_pred_probs = model.predict(X_test, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Final Test Accuracy: {test_acc:.4f}")
print(f"Final Test Loss: {test_loss:.4f}")

print("\n=== CLASSIFICATION REPORT ===")
print(classification_report(y_test, y_pred, digits=3))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7.5, 6.5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(10), yticklabels=range(10))
plt.xlabel('Predicted digit')
plt.ylabel('Actual digit')
plt.title(f'Confusion Matrix — Final CNN\nTest Accuracy = {test_acc:.2%}', fontweight='bold')
plt.tight_layout()
plt.savefig('fig4_confusion_matrix.png', dpi=120)
plt.close()

# Misclassified examples
misclassified_idx = np.where(y_pred != y_test)[0]
print(f"\nMisclassified: {len(misclassified_idx)} out of {len(y_test)}")

n_show = min(8, len(misclassified_idx))
if n_show > 0:
    fig, axes = plt.subplots(1, n_show, figsize=(2*n_show, 2.5))
    if n_show == 1:
        axes = [axes]
    for i, idx in enumerate(misclassified_idx[:n_show]):
        axes[i].imshow(X_test[idx].reshape(8,8), cmap='gray_r')
        conf = y_pred_probs[idx][y_pred[idx]]
        axes[i].set_title(f"True: {y_test[idx]}\nPred: {y_pred[idx]} ({conf:.0%})", fontsize=9)
        axes[i].axis('off')
    plt.suptitle('Misclassified Test Examples', fontweight='bold')
    plt.tight_layout()
    plt.savefig('fig5_misclassified.png', dpi=120)
    plt.close()
    print("Saved misclassified examples figure.")
else:
    print("No misclassified examples to show.")
