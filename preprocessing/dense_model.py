import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from keras import layers, models, optimizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout, Input # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore


def create_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # Helps prevent overfitting
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),  # Helps prevent overfitting
        layers.Dense(num_classes, activation='softmax')
    ])
    return model

input_shape = (42,)  # Shape of your input features
num_classes = 29  # Updated number of unique labels

model = create_model(input_shape, num_classes)

adam_opt = optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=adam_opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

data_dict = pickle.load(open('./data.pickle', 'rb'))

max_length = max(len(seq) for seq in data_dict['data'])
padded_data = [seq + [0] * (max_length - len(seq)) for seq in data_dict['data']]

data = np.asarray(padded_data)
labels = np.asarray(data_dict['labels'])
label_encoder = LabelEncoder()
integer_encoded_labels = label_encoder.fit_transform(labels)
with open('label_encoder.pkl', 'wb') as file:
    pickle.dump(label_encoder, file)

x_train, x_val, y_train, y_val = train_test_split(data, integer_encoded_labels, test_size=0.2, shuffle=True, stratify=labels)

history = model.fit(x_train, y_train,
                    epochs=50,  # Adjust based on when you see the validation loss plateau
                    batch_size=32,  # Typical batch size, adjust as needed
                    validation_data=(x_val, y_val),
                    callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)])

model.save('my_model')

# Access the history dictionary
training_loss = history.history['loss']
training_accuracy = history.history['accuracy']
validation_loss = history.history['val_loss']
validation_accuracy = history.history['val_accuracy']

# Make predictions
predictions = model.predict(x_val)
predicted_classes = np.argmax(predictions, axis=1)

# Generate the confusion matrix
cm = confusion_matrix(y_val, predicted_classes)

# Plot the confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.ylabel('Actual Labels')
plt.xlabel('Predicted Labels')
plt.show()

# Plotting training and validation loss
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

# Plotting training and validation accuracy
plt.subplot(1, 2, 2)
plt.plot(training_accuracy, label='Training Accuracy')
plt.plot(validation_accuracy, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()



