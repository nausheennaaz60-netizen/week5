import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

data = np.load('digits_split.npz')
X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

print("Train:", X_train.shape, "Val:", X_val.shape, "Test:", X_test.shape)

# ============ MODEL A: CNN WITHOUT REGULARIZATION (to demonstrate overfitting) ============
def build_overfit_cnn():
    model = keras.Sequential([
        layers.Input(shape=(8, 8, 1)),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.MaxPooling2D((2,2)),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(10, activation='softmax')
    ], name='overfit_cnn')
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model_overfit = build_overfit_cnn()
model_overfit.summary()
print(f"\nTotal params (overfit model): {model_overfit.count_params():,}")

history_overfit = model_overfit.fit(
    X_train, y_train, validation_data=(X_val, y_val),
    epochs=60, batch_size=32, verbose=0
)

# ============ MODEL B: FINAL CNN WITH REGULARIZATION ============
def build_final_cnn():
    model = keras.Sequential([
        layers.Input(shape=(8, 8, 1)),
        layers.Conv2D(16, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(10, activation='softmax')
    ], name='final_cnn')
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

model_final = build_final_cnn()
model_final.summary()
print(f"\nTotal params (final model): {model_final.count_params():,}")

# Data augmentation: small random shifts/rotations, appropriate for digit images
# (kept mild since digits are position/orientation sensitive - e.g. 6 vs 9)
datagen = keras.preprocessing.image.ImageDataGenerator(
    rotation_range=8, width_shift_range=0.08, height_shift_range=0.08, zoom_range=0.08
)
datagen.fit(X_train)

early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=10, restore_best_weights=True
)

history_final = model_final.fit(
    datagen.flow(X_train, y_train, batch_size=32),
    validation_data=(X_val, y_val),
    epochs=100, callbacks=[early_stop], verbose=0
)

print(f"\nFinal model stopped at epoch {len(history_final.history['loss'])} (early stopping)")

# ============ COMPARISON PLOT: OVERFITTING DEMONSTRATION ============
fig, axes = plt.subplots(2, 2, figsize=(12, 9))

axes[0,0].plot(history_overfit.history['accuracy'], label='Train', color='#E07A5F')
axes[0,0].plot(history_overfit.history['val_accuracy'], label='Validation', color='#3D5A80')
axes[0,0].set_title('Model A (No Regularization): Accuracy', fontweight='bold')
axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Accuracy'); axes[0,0].legend()

axes[0,1].plot(history_overfit.history['loss'], label='Train', color='#E07A5F')
axes[0,1].plot(history_overfit.history['val_loss'], label='Validation', color='#3D5A80')
axes[0,1].set_title('Model A (No Regularization): Loss', fontweight='bold')
axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('Loss'); axes[0,1].legend()

axes[1,0].plot(history_final.history['accuracy'], label='Train', color='#E07A5F')
axes[1,0].plot(history_final.history['val_accuracy'], label='Validation', color='#3D5A80')
axes[1,0].set_title('Model B (Regularized + Aug): Accuracy', fontweight='bold')
axes[1,0].set_xlabel('Epoch'); axes[1,0].set_ylabel('Accuracy'); axes[1,0].legend()

axes[1,1].plot(history_final.history['loss'], label='Train', color='#E07A5F')
axes[1,1].plot(history_final.history['val_loss'], label='Validation', color='#3D5A80')
axes[1,1].set_title('Model B (Regularized + Aug): Loss', fontweight='bold')
axes[1,1].set_xlabel('Epoch'); axes[1,1].set_ylabel('Loss'); axes[1,1].legend()

plt.tight_layout()
plt.savefig('fig3_training_curves.png', dpi=120)
plt.close()

# Evaluate both on test set
test_loss_a, test_acc_a = model_overfit.evaluate(X_test, y_test, verbose=0)
test_loss_b, test_acc_b = model_final.evaluate(X_test, y_test, verbose=0)
train_acc_a = history_overfit.history['accuracy'][-1]
train_acc_b = history_final.history['accuracy'][-1]

print(f"\nModel A (no regularization) -- Final Train Acc: {train_acc_a:.4f}, Test Acc: {test_acc_a:.4f}, Gap: {train_acc_a-test_acc_a:.4f}")
print(f"Model B (regularized)       -- Final Train Acc: {train_acc_b:.4f}, Test Acc: {test_acc_b:.4f}, Gap: {train_acc_b-test_acc_b:.4f}")

model_final.save('final_cnn_model.keras')
np.save('test_acc_comparison.npy', {'model_a': (train_acc_a, test_acc_a), 'model_b': (train_acc_b, test_acc_b)})
print("\nModel saved.")
