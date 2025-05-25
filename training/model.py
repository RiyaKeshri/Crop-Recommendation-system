import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv("crop_data1.csv")  # Replace with your CSV file path

# Separate features and labels
X = data[['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']].values
y = data['label']

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reshape data for CNN (convert tabular data into 2D format)
X_cnn = X_scaled.reshape(-1, 7, 1, 1)  # (samples, features, height, channels)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_cnn, y_encoded, test_size=0.2, random_state=42)

# Convert labels to one-hot encoding for CNN
y_train_cnn = to_categorical(y_train)
y_test_cnn = to_categorical(y_test)

# Build the CNN model
model = Sequential([
    Conv2D(16, kernel_size=(2, 1), activation='relu', input_shape=(7, 1, 1)),
    MaxPooling2D(pool_size=(2, 1)),
    Dropout(0.2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.3),
    Dense(y_train_cnn.shape[1], activation='softmax')  # Number of output classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the CNN model
history = model.fit(X_train, y_train_cnn, validation_split=0.2, epochs=50, batch_size=16, verbose=1)

# Evaluate the model
cnn_loss, cnn_accuracy = model.evaluate(X_test, y_test_cnn, verbose=0)
print(f"CNN Test Accuracy: {cnn_accuracy:.2f}")

# Save the model
model.save("cnn_model.h5")
print("CNN model saved as 'cnn_model.h5'.")

# Plot training accuracy and loss
plt.figure(figsize=(12, 6))

# Accuracy plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy', color='blue')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy', color='orange')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Loss plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss', color='blue')
plt.plot(history.history['val_loss'], label='Validation Loss', color='orange')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

# Save the plots
plt.tight_layout()
plt.savefig("training_accuracy_loss.png")
plt.show()
