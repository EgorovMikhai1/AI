import os

import numpy as np
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential


def read_sequence_from_file(filename):
    with open(filename, 'r') as file:
        sequence = [float(line.strip()) for line in file if line.strip()]
    return sequence


def create_dataset(sequence, n_steps):
    X, y = [], []
    for i in range(len(sequence)):
        end_ix = i + n_steps
        if end_ix > len(sequence) - 1:
            break
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return np.array(X), np.array(y)


def build_model(n_steps):
    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(n_steps, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model


def predict_and_correct(model, n_steps, sequence, true_sequence, start_idx):
    while start_idx < len(true_sequence):
        x_input = np.array(sequence[-n_steps:]).reshape((1, n_steps, 1))
        yhat = model.predict(x_input, verbose=0)
        predicted_number = yhat[0][0]
        true_number = true_sequence[start_idx]

        print(f"Predicted next number: {predicted_number}")
        print(f"True next number: {true_number}")

        if predicted_number != true_number:
            print("Prediction was incorrect. Correcting the model...")
            sequence.append(true_number)
            X, y = create_dataset(sequence, n_steps)
            X = X.reshape((X.shape[0], X.shape[1], 1))
            model.fit(X, y, epochs=5, verbose=1)
        else:
            print("Prediction was correct!")
            sequence.append(predicted_number)

        start_idx += 1


def train_model_on_initial_sequence(filename, n_steps, epochs=20):
    # Read sequence from file
    sequence = read_sequence_from_file(filename)

    # Create dataset
    X, y = create_dataset(sequence[:10000], n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build model
    model = build_model(n_steps)

    # Fit model
    model.fit(X, y, epochs=epochs, verbose=1)

    return model, sequence


# Main code
filename = 'sequence.txt'
n_steps = 10

# Check if file exists, if not create it
if not os.path.isfile(filename):
    with open(filename, 'w') as file:
        for i in range(10000):
            file.write(f"{i}\n")
    print(f"File '{filename}' created successfully.")

# Train model on the initial sequence
model, sequence = train_model_on_initial_sequence(filename, n_steps)

# Predict and correct
true_sequence = read_sequence_from_file(filename)
predict_and_correct(model, n_steps, sequence[:10000], true_sequence, 10000)
