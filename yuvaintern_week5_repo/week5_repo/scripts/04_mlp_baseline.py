import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import warnings
warnings.filterwarnings('ignore')

np.random.seed(42)
tf.random.set_seed(42)

data = np.load('digits_split.npz')
X_train, X_val, X_test = data['X_train'], data['X_val'], data['X_test']
y_train, y_val, y_test = data['y_train'], data['y_val'], data['y_test']

# Simple dense (MLP) baseline, no convolution - to justify why CNN architecture was chosen
mlp = keras.Sequential([
    layers.Input(shape=(8,8,1)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(32, activation='relu'),
    layers.Dense(10, activation='softmax')
], name='mlp_baseline')
mlp.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print(f"MLP baseline params: {mlp.count_params():,}")

early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
hist_mlp = mlp.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=100, callbacks=[early_stop], verbose=0)

test_loss, test_acc = mlp.evaluate(X_test, y_test, verbose=0)
print(f"MLP baseline test accuracy: {test_acc:.4f} (stopped at epoch {len(hist_mlp.history['loss'])})")
