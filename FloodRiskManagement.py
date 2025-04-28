from netCDF4 import Dataset
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Load the dataset
dataset = Dataset('sample_data/NCALDAS_NOAH0125.A20161228.002.nc4', mode='r')

rainf = dataset.variables['Rainf'][0]          # Rainfall
streamflow = dataset.variables['Streamflow'][0]  # Streamflow
flood_frac = dataset.variables['FloodedFrac'][0] # Flooded fraction


X = np.stack([rainf.flatten(), streamflow.flatten()], axis=1)


y = (flood_frac.flatten() > 0).astype(int)


mask = ~np.isnan(X).any(axis=1)
X = X[mask]
y = y[mask]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = Sequential([
    Dense(32, activation='relu', input_shape=(X.shape[1],)),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')
])


model.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

y_pred = (model.predict(X_test) > 0.5).astype(int)

y_test_np = np.array(y_test)
y_pred_np = np.array(y_pred)

print("Accuracy:", accuracy_score(y_test_np, y_pred_np))
print("Confusion Matrix:\n", confusion_matrix(y_test_np, y_pred_np))

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.title("Loss")

plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend(); plt.title("Accuracy")
plt.show()
