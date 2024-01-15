import numpy as np
import pandas as pd
import ast
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

data = pd.read_excel("/Users/shauryajain/alien-rematch/project3/AI-Project3-Code/Model1 data.xlsx")
data.drop(4060)
ship = data['layout of ship ']
alien_prob = data['alien prob']
crew_prob = data['crew prob']
bot = data['position of bot']
move = data["next move"].tolist()
print("Data size: ", len(data))

X_train = []
y_train = []
X_test = []
y_test = []

for i in range(len(data)-1):
  print(i)
  a = ast.literal_eval(ship[i])
  b = list(ast.literal_eval(alien_prob[i][1:-1]))
  c = list(ast.literal_eval(crew_prob[i][1:-1]))
  d = list(ast.literal_eval(bot[i]))
  print(bot[i])
  print(d)

  X_train.append(a+b+c+d)
  y_train.append(float(move[i]))

X_train = np.array(X_train)
y_train = np.array(y_train)
y_train_cat = to_categorical(y_train, num_classes=4)

def yeo_johnson(data):
    lambda_ = np.percentile(data, 75) - np.percentile(data, 25)
    if lambda_ > 0:
        return np.log(data + lambda_)
    else:
        return -np.log(-data + lambda_ + 1)

# Assuming `data` is your 900 columns data
# Step 1: Data Preprocessing
data_transformed = np.apply_along_axis(yeo_johnson, 1, X_train[:,2500:5000])

# Step 2: Normalization
min_values = np.min(data_transformed, axis=0)
max_values = np.max(data_transformed, axis=0)
data_scaled = (data_transformed - min_values) / (max_values - min_values)

# Assuming you have a dataset (X, y) and you want to split it into training and validation sets
X_train_enc, X_val_enc, y_train_enc, y_val_enc = train_test_split(data_scaled, data_scaled, test_size=0.1, random_state=42)
# Activation functions and their derivatives


def binary_crossentropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss

# Initialize weights and biases
input_size = data_scaled.shape[1]
hidden_size_1 = 256
hidden_size_2 = 64
hidden_size_3 = 25
hidden_size_4 = 64
hidden_size_5 = 256
output_size = data_scaled.shape[1]

np.random.seed(42)

weights_input_hidden_1 = np.random.rand(input_size, hidden_size_1)
biases_hidden_1 = np.zeros((1, hidden_size_1))

weights_hidden_1_hidden_2 = np.random.rand(hidden_size_1, hidden_size_2)
biases_hidden_2 = np.zeros((1, hidden_size_2))

weights_hidden_2_hidden_3 = np.random.rand(hidden_size_2, hidden_size_3)
biases_hidden_3 = np.zeros((1, hidden_size_3))

weights_hidden_3_hidden_4 = np.random.rand(hidden_size_3, hidden_size_4)
biases_hidden_4 = np.zeros((1, hidden_size_4))

weights_hidden_4_hidden_5 = np.random.rand(hidden_size_4, hidden_size_5)
biases_hidden_5 = np.zeros((1, hidden_size_5))

weights_hidden_5_output = np.random.rand(hidden_size_5, output_size)
biases_output = np.zeros((1, output_size))

# Training parameters
learning_rate = 0.1
epochs = 25
batch_size = 128

# Training loop with backpropagation for the provided architecture
for epoch in range(epochs):
    total_loss = 0
    correct_predictions = 0

    # Training
    for i in range(0, len(X_train_enc), batch_size):
        batch_X = X_train_enc[i:i + batch_size]
        batch_y = y_train_enc[i:i + batch_size]

        # Forward pass
        hidden_layer_1_input = np.dot(batch_X, weights_input_hidden_1) + biases_hidden_1
        hidden_layer_1_output = relu(hidden_layer_1_input)

        hidden_layer_2_input = np.dot(hidden_layer_1_output, weights_hidden_1_hidden_2) + biases_hidden_2
        hidden_layer_2_output = relu(hidden_layer_2_input)

        hidden_layer_3_input = np.dot(hidden_layer_2_output, weights_hidden_2_hidden_3) + biases_hidden_3
        hidden_layer_3_output = relu(hidden_layer_3_input)

        hidden_layer_4_input = np.dot(hidden_layer_3_output, weights_hidden_3_hidden_4) + biases_hidden_4
        hidden_layer_4_output = relu(hidden_layer_4_input)

        hidden_layer_5_input = np.dot(hidden_layer_4_output, weights_hidden_4_hidden_5) + biases_hidden_5
        hidden_layer_5_output = relu(hidden_layer_5_input)

        output_layer_input = np.dot(hidden_layer_5_output, weights_hidden_5_output) + biases_output
        output_layer_output = output_layer_input

        # Compute loss
        loss = binary_crossentropy(batch_y, output_layer_output)
        total_loss += loss

        # Compute accuracy
        predictions = output_layer_output
        true_labels = batch_y
        correct_predictions += np.sum(predictions == true_labels)

        # Backward pass
        output_error = output_layer_output - batch_y
        hidden_layer_5_error = np.dot(output_error, weights_hidden_5_output.T) * relu_derivative(hidden_layer_5_output)
        hidden_layer_4_error = np.dot(hidden_layer_5_error, weights_hidden_4_hidden_5.T) * relu_derivative(hidden_layer_4_output)
        hidden_layer_3_error = np.dot(hidden_layer_4_error, weights_hidden_3_hidden_4.T) * relu_derivative(hidden_layer_3_output)
        hidden_layer_2_error = np.dot(hidden_layer_3_error, weights_hidden_2_hidden_3.T) * relu_derivative(hidden_layer_2_output)
        hidden_layer_1_error = np.dot(hidden_layer_2_error, weights_hidden_1_hidden_2.T) * relu_derivative(hidden_layer_1_output)

        # Update weights and biases using gradient descent
        weights_hidden_5_output -= learning_rate * np.dot(hidden_layer_5_output.T, output_error)
        biases_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True)

        weights_hidden_4_hidden_5 -= learning_rate * np.dot(hidden_layer_4_output.T, hidden_layer_5_error)
        biases_hidden_5 -= learning_rate * np.sum(hidden_layer_5_error, axis=0, keepdims=True)

        weights_hidden_3_hidden_4 -= learning_rate * np.dot(hidden_layer_3_output.T, hidden_layer_4_error)
        biases_hidden_4 -= learning_rate * np.sum(hidden_layer_4_error, axis=0, keepdims=True)

        weights_hidden_2_hidden_3 -= learning_rate * np.dot(hidden_layer_2_output.T, hidden_layer_3_error)
        biases_hidden_3 -= learning_rate * np.sum(hidden_layer_3_error, axis=0, keepdims=True)

        weights_hidden_1_hidden_2 -= learning_rate * np.dot(hidden_layer_1_output.T, hidden_layer_2_error)
        biases_hidden_2 -= learning_rate * np.sum(hidden_layer_2_error, axis=0, keepdims=True)

        weights_input_hidden_1 -= learning_rate * np.dot(batch_X.T, hidden_layer_1_error)
        biases_hidden_1 -= learning_rate * np.sum(hidden_layer_1_error, axis=0, keepdims=True)

    # Validation
    val_total_loss = 0
    val_correct_predictions = 0

    for i in range(0, len(X_val_enc), batch_size):
        val_batch_X = X_val_enc[i:i + batch_size]
        val_batch_y = y_val_enc[i:i + batch_size]

        # Forward pass for validation
        hidden_layer_1_input_val = np.dot(val_batch_X, weights_input_hidden_1) + biases_hidden_1
        hidden_layer_1_output_val = relu(hidden_layer_1_input_val)

        hidden_layer_2_input_val = np.dot(hidden_layer_1_output_val, weights_hidden_1_hidden_2) + biases_hidden_2
        hidden_layer_2_output_val = relu(hidden_layer_2_input_val)

        hidden_layer_3_input_val = np.dot(hidden_layer_2_output_val, weights_hidden_2_hidden_3) + biases_hidden_3
        hidden_layer_3_output_val = relu(hidden_layer_3_input_val)

        hidden_layer_4_input_val = np.dot(hidden_layer_3_output_val, weights_hidden_3_hidden_4) + biases_hidden_4
        hidden_layer_4_output_val = relu(hidden_layer_4_input_val)

        hidden_layer_5_input_val = np.dot(hidden_layer_4_output_val, weights_hidden_4_hidden_5) + biases_hidden_5
        hidden_layer_5_output_val = relu(hidden_layer_5_input_val)

        output_layer_input_val = np.dot(hidden_layer_5_output_val, weights_hidden_5_output) + biases_output
        output_layer_output_val = output_layer_input_val

        # Compute validation loss
        val_loss = binary_crossentropy(val_batch_y, output_layer_output_val)
        val_total_loss += val_loss

        # Compute validation accuracy
        val_predictions = output_layer_output_val
        val_true_labels = val_batch_y
        val_correct_predictions += np.sum(val_predictions == val_true_labels)

    # Calculate and print average training and validation loss and accuracy for every epoch
    average_loss = total_loss / (len(X_train_enc) / batch_size)
    training_accuracy = correct_predictions / len(X_train_enc)

    val_average_loss = val_total_loss / (len(X_val_enc) / batch_size)
    val_accuracy = val_correct_predictions / len(X_val_enc)

    print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {average_loss}, Validation Loss: {val_average_loss}\n')

# Evaluate the model on the test set
hidden_layer_1_input_test = np.dot(data_scaled, weights_input_hidden_1) + biases_hidden_1
hidden_layer_1_output_test = relu(hidden_layer_1_input_test)

hidden_layer_2_input_test = np.dot(hidden_layer_1_output_test, weights_hidden_1_hidden_2) + biases_hidden_2
hidden_layer_2_output_test = relu(hidden_layer_2_input_test)

hidden_layer_3_input_test = np.dot(hidden_layer_2_output_test, weights_hidden_2_hidden_3) + biases_hidden_3
hidden_layer_3_output_test = relu(hidden_layer_3_input_test)

data_reduced = hidden_layer_3_output_test
df_belief_network_alien = pd.DataFrame(data_reduced)

# Step 1: Data Preprocessing
data_transformed = np.apply_along_axis(yeo_johnson, 1, X_train[:,5000:7500])

# Step 2: Normalization
min_values = np.min(data_transformed, axis=0)
max_values = np.max(data_transformed, axis=0)
data_scaled = (data_transformed - min_values) / (max_values - min_values)

# Assuming you have a dataset (X, y) and you want to split it into training and validation sets
X_train_enc, X_val_enc, y_train_enc, y_val_enc = train_test_split(data_scaled, data_scaled, test_size=0.1, random_state=42)
# Activation functions and their derivatives


# Initialize weights and biases
input_size = data_scaled.shape[1]
hidden_size_1 = 256
hidden_size_2 = 64
hidden_size_3 = 25
hidden_size_4 = 64
hidden_size_5 = 256
output_size = data_scaled.shape[1]

np.random.seed(42)

weights_input_hidden_1 = np.random.rand(input_size, hidden_size_1)
biases_hidden_1 = np.zeros((1, hidden_size_1))

weights_hidden_1_hidden_2 = np.random.rand(hidden_size_1, hidden_size_2)
biases_hidden_2 = np.zeros((1, hidden_size_2))

weights_hidden_2_hidden_3 = np.random.rand(hidden_size_2, hidden_size_3)
biases_hidden_3 = np.zeros((1, hidden_size_3))

weights_hidden_3_hidden_4 = np.random.rand(hidden_size_3, hidden_size_4)
biases_hidden_4 = np.zeros((1, hidden_size_4))

weights_hidden_4_hidden_5 = np.random.rand(hidden_size_4, hidden_size_5)
biases_hidden_5 = np.zeros((1, hidden_size_5))

weights_hidden_5_output = np.random.rand(hidden_size_5, output_size)
biases_output = np.zeros((1, output_size))

# Training parameters
learning_rate = 0.1
epochs = 25
batch_size = 128

# Training loop with backpropagation for the provided architecture
for epoch in range(epochs):
    total_loss = 0
    correct_predictions = 0

    # Training
    for i in range(0, len(X_train_enc), batch_size):
        batch_X = X_train_enc[i:i + batch_size]
        batch_y = y_train_enc[i:i + batch_size]

        # Forward pass
        hidden_layer_1_input = np.dot(batch_X, weights_input_hidden_1) + biases_hidden_1
        hidden_layer_1_output = relu(hidden_layer_1_input)

        hidden_layer_2_input = np.dot(hidden_layer_1_output, weights_hidden_1_hidden_2) + biases_hidden_2
        hidden_layer_2_output = relu(hidden_layer_2_input)

        hidden_layer_3_input = np.dot(hidden_layer_2_output, weights_hidden_2_hidden_3) + biases_hidden_3
        hidden_layer_3_output = relu(hidden_layer_3_input)

        hidden_layer_4_input = np.dot(hidden_layer_3_output, weights_hidden_3_hidden_4) + biases_hidden_4
        hidden_layer_4_output = relu(hidden_layer_4_input)

        hidden_layer_5_input = np.dot(hidden_layer_4_output, weights_hidden_4_hidden_5) + biases_hidden_5
        hidden_layer_5_output = relu(hidden_layer_5_input)

        output_layer_input = np.dot(hidden_layer_5_output, weights_hidden_5_output) + biases_output
        output_layer_output = output_layer_input

        # Compute loss
        loss = binary_crossentropy(batch_y, output_layer_output)
        total_loss += loss

        # Compute accuracy
        predictions = output_layer_output
        true_labels = batch_y
        correct_predictions += np.sum(predictions == true_labels)

        # Backward pass
        output_error = output_layer_output - batch_y
        hidden_layer_5_error = np.dot(output_error, weights_hidden_5_output.T) * relu_derivative(hidden_layer_5_output)
        hidden_layer_4_error = np.dot(hidden_layer_5_error, weights_hidden_4_hidden_5.T) * relu_derivative(hidden_layer_4_output)
        hidden_layer_3_error = np.dot(hidden_layer_4_error, weights_hidden_3_hidden_4.T) * relu_derivative(hidden_layer_3_output)
        hidden_layer_2_error = np.dot(hidden_layer_3_error, weights_hidden_2_hidden_3.T) * relu_derivative(hidden_layer_2_output)
        hidden_layer_1_error = np.dot(hidden_layer_2_error, weights_hidden_1_hidden_2.T) * relu_derivative(hidden_layer_1_output)

        # Update weights and biases using gradient descent
        weights_hidden_5_output -= learning_rate * np.dot(hidden_layer_5_output.T, output_error)
        biases_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True)

        weights_hidden_4_hidden_5 -= learning_rate * np.dot(hidden_layer_4_output.T, hidden_layer_5_error)
        biases_hidden_5 -= learning_rate * np.sum(hidden_layer_5_error, axis=0, keepdims=True)

        weights_hidden_3_hidden_4 -= learning_rate * np.dot(hidden_layer_3_output.T, hidden_layer_4_error)
        biases_hidden_4 -= learning_rate * np.sum(hidden_layer_4_error, axis=0, keepdims=True)

        weights_hidden_2_hidden_3 -= learning_rate * np.dot(hidden_layer_2_output.T, hidden_layer_3_error)
        biases_hidden_3 -= learning_rate * np.sum(hidden_layer_3_error, axis=0, keepdims=True)

        weights_hidden_1_hidden_2 -= learning_rate * np.dot(hidden_layer_1_output.T, hidden_layer_2_error)
        biases_hidden_2 -= learning_rate * np.sum(hidden_layer_2_error, axis=0, keepdims=True)

        weights_input_hidden_1 -= learning_rate * np.dot(batch_X.T, hidden_layer_1_error)
        biases_hidden_1 -= learning_rate * np.sum(hidden_layer_1_error, axis=0, keepdims=True)

    # Validation
    val_total_loss = 0
    val_correct_predictions = 0

    for i in range(0, len(X_val_enc), batch_size):
        val_batch_X = X_val_enc[i:i + batch_size]
        val_batch_y = y_val_enc[i:i + batch_size]

        # Forward pass for validation
        hidden_layer_1_input_val = np.dot(val_batch_X, weights_input_hidden_1) + biases_hidden_1
        hidden_layer_1_output_val = relu(hidden_layer_1_input_val)

        hidden_layer_2_input_val = np.dot(hidden_layer_1_output_val, weights_hidden_1_hidden_2) + biases_hidden_2
        hidden_layer_2_output_val = relu(hidden_layer_2_input_val)

        hidden_layer_3_input_val = np.dot(hidden_layer_2_output_val, weights_hidden_2_hidden_3) + biases_hidden_3
        hidden_layer_3_output_val = relu(hidden_layer_3_input_val)

        hidden_layer_4_input_val = np.dot(hidden_layer_3_output_val, weights_hidden_3_hidden_4) + biases_hidden_4
        hidden_layer_4_output_val = relu(hidden_layer_4_input_val)

        hidden_layer_5_input_val = np.dot(hidden_layer_4_output_val, weights_hidden_4_hidden_5) + biases_hidden_5
        hidden_layer_5_output_val = relu(hidden_layer_5_input_val)

        output_layer_input_val = np.dot(hidden_layer_5_output_val, weights_hidden_5_output) + biases_output
        output_layer_output_val = output_layer_input_val

        # Compute validation loss
        val_loss = binary_crossentropy(val_batch_y, output_layer_output_val)
        val_total_loss += val_loss

        # Compute validation accuracy
        val_predictions = output_layer_output_val
        val_true_labels = val_batch_y
        val_correct_predictions += np.sum(val_predictions == val_true_labels)

    # Calculate and print average training and validation loss and accuracy for every epoch
    average_loss = total_loss / (len(X_train_enc) / batch_size)
    training_accuracy = correct_predictions / len(X_train_enc)

    val_average_loss = val_total_loss / (len(X_val_enc) / batch_size)
    val_accuracy = val_correct_predictions / len(X_val_enc)

    print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {average_loss}, Validation Loss: {val_average_loss}\n')

# Evaluate the model on the test set
hidden_layer_1_input_test = np.dot(data_scaled, weights_input_hidden_1) + biases_hidden_1
hidden_layer_1_output_test = relu(hidden_layer_1_input_test)

hidden_layer_2_input_test = np.dot(hidden_layer_1_output_test, weights_hidden_1_hidden_2) + biases_hidden_2
hidden_layer_2_output_test = relu(hidden_layer_2_input_test)

hidden_layer_3_input_test = np.dot(hidden_layer_2_output_test, weights_hidden_2_hidden_3) + biases_hidden_3
hidden_layer_3_output_test = relu(hidden_layer_3_input_test)

data_reduced = hidden_layer_3_output_test
df_belief_network_crew = pd.DataFrame(data_reduced)

alien_enc = np.array(df_belief_network_alien)
crew_enc = np.array(df_belief_network_crew)

X_tr_move = np.load("/content/drive/MyDrive/AI-Project-3-data/old data/final npy/pos_moves_X_train.npy")
bpos_train = X_train[:,-4:-2]
X_train_fin = np.concatenate((alien_enc,crew_enc,bpos_train[:alien_enc.shape[0]],X_tr_move[:alien_enc.shape[0]]), axis = -1)

# Assuming you have a dataset (X, y) and you want to split it into training and validation sets
X_train_fin, X_test_fin, y_train_cat, y_test_cat = train_test_split(X_train_fin, y_train_cat, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_fin, y_train_cat, test_size=0.1, random_state=42)


# Softmax activation function with a mask
def softmax(x, mask_in):
    x_norm = x - np.max(x, axis=-1, keepdims=True)
    exp_values = np.exp(x_norm)
    exp_values = np.multiply(exp_values, mask_in)
    sum_exp = np.sum(exp_values, axis=-1, keepdims=True) + 1e-10  # Add small constant to prevent division by zero
    probabilities = exp_values / sum_exp
    return probabilities

# Categorical crossentropy loss function
def categorical_crossentropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    loss = -np.sum(y_true * np.log(y_pred)) / len(y_true)
    return loss


# Initialize weights and biases
input_size = X_train_fin.shape[1]
hidden_size_1 = 32
hidden_size_2 = 16
hidden_size_3 = 8
hidden_size_4 = 4
output_size = 4

np.random.seed(42)

weights_input_hidden_1 = np.random.rand(input_size, hidden_size_1)
biases_hidden_1 = np.zeros((1, hidden_size_1))

weights_hidden_1_hidden_2 = np.random.rand(hidden_size_1, hidden_size_2)
biases_hidden_2 = np.zeros((1, hidden_size_2))

weights_hidden_2_hidden_3 = np.random.rand(hidden_size_2, hidden_size_3)
biases_hidden_3 = np.zeros((1, hidden_size_3))

weights_hidden_3_hidden_4 = np.random.rand(hidden_size_3, hidden_size_4)
biases_hidden_4 = np.zeros((1, hidden_size_4))

weights_hidden_4_output = np.random.rand(hidden_size_4, output_size)
biases_output = np.zeros((1, output_size))

# Training parameters
learning_rate = 0.05
epochs = 50
batch_size = 128

# Training loop with backpropagation for the provided architecture
for epoch in range(epochs):
    total_loss = 0
    correct_predictions = 0

    # Training
    for i in range(0, len(X_train), batch_size):
        batch_X = X_train[i:i + batch_size]
        batch_y = y_train[i:i + batch_size]

        # Forward pass
        hidden_layer_1_input = np.dot(batch_X, weights_input_hidden_1) + biases_hidden_1
        hidden_layer_1_output = relu(hidden_layer_1_input)

        hidden_layer_2_input = np.dot(hidden_layer_1_output, weights_hidden_1_hidden_2) + biases_hidden_2
        hidden_layer_2_output = relu(hidden_layer_2_input)

        hidden_layer_3_input = np.dot(hidden_layer_2_output, weights_hidden_2_hidden_3) + biases_hidden_3
        hidden_layer_3_output = relu(hidden_layer_3_input)

        hidden_layer_4_input = np.dot(hidden_layer_3_output, weights_hidden_3_hidden_4) + biases_hidden_4
        hidden_layer_4_output = relu(hidden_layer_4_input)

        output_layer_input = np.dot(hidden_layer_4_output, weights_hidden_4_output) + biases_output
        output_layer_output = softmax(output_layer_input, batch_X[:,-4:])

        # Compute loss
        loss = categorical_crossentropy(batch_y, output_layer_output)
        total_loss += loss

        # Compute accuracy
        predictions = np.argmax(output_layer_output, axis=1)
        true_labels = np.argmax(batch_y, axis=1)
        correct_predictions += np.sum(predictions == true_labels)

        # Backward pass
        output_error = output_layer_output - batch_y
        hidden_layer_4_error = np.dot(output_error, weights_hidden_4_output.T) * relu_derivative(hidden_layer_4_output)
        hidden_layer_3_error = np.dot(hidden_layer_4_error, weights_hidden_3_hidden_4.T) * relu_derivative(hidden_layer_3_output)
        hidden_layer_2_error = np.dot(hidden_layer_3_error, weights_hidden_2_hidden_3.T) * relu_derivative(hidden_layer_2_output)
        hidden_layer_1_error = np.dot(hidden_layer_2_error, weights_hidden_1_hidden_2.T) * relu_derivative(hidden_layer_1_output)

        # Update weights and biases using gradient descent
        weights_hidden_4_output -= learning_rate * np.dot(hidden_layer_4_output.T, output_error)
        biases_output -= learning_rate * np.sum(output_error, axis=0, keepdims=True)

        weights_hidden_3_hidden_4 -= learning_rate * np.dot(hidden_layer_3_output.T, hidden_layer_4_error)
        biases_hidden_4 -= learning_rate * np.sum(hidden_layer_4_error, axis=0, keepdims=True)


        weights_hidden_2_hidden_3 -= learning_rate * np.dot(hidden_layer_2_output.T, hidden_layer_3_error)
        biases_hidden_3 -= learning_rate * np.sum(hidden_layer_3_error, axis=0, keepdims=True)

        weights_hidden_1_hidden_2 -= learning_rate * np.dot(hidden_layer_1_output.T, hidden_layer_2_error)
        biases_hidden_2 -= learning_rate * np.sum(hidden_layer_2_error, axis=0, keepdims=True)

        weights_input_hidden_1 -= learning_rate * np.dot(batch_X.T, hidden_layer_1_error)
        biases_hidden_1 -= learning_rate * np.sum(hidden_layer_1_error, axis=0, keepdims=True)

    # Validation
    val_total_loss = 0
    val_correct_predictions = 0

    for i in range(0, len(X_val), batch_size):
        val_batch_X = X_val[i:i + batch_size]
        val_batch_y = y_val[i:i + batch_size]

        # Forward pass for validation
        hidden_layer_1_input_val = np.dot(val_batch_X, weights_input_hidden_1) + biases_hidden_1
        hidden_layer_1_output_val = relu(hidden_layer_1_input_val)

        hidden_layer_2_input_val = np.dot(hidden_layer_1_output_val, weights_hidden_1_hidden_2) + biases_hidden_2
        hidden_layer_2_output_val = relu(hidden_layer_2_input_val)

        hidden_layer_3_input_val = np.dot(hidden_layer_2_output_val, weights_hidden_2_hidden_3) + biases_hidden_3
        hidden_layer_3_output_val = relu(hidden_layer_3_input_val)

        hidden_layer_4_input_val = np.dot(hidden_layer_3_output_val, weights_hidden_3_hidden_4) + biases_hidden_4
        hidden_layer_4_output_val = relu(hidden_layer_4_input_val)

        output_layer_input_val = np.dot(hidden_layer_4_output_val, weights_hidden_4_output) + biases_output
        output_layer_output_val = softmax(output_layer_input_val, val_batch_X[:,-4:])

        # Compute validation loss
        val_loss = categorical_crossentropy(val_batch_y, output_layer_output_val)
        val_total_loss += val_loss

        # Compute validation accuracy
        val_predictions = np.argmax(output_layer_output_val, axis=1)
        val_true_labels = np.argmax(val_batch_y, axis=1)
        val_correct_predictions += np.sum(val_predictions == val_true_labels)

    # Calculate and print average training and validation loss and accuracy for every epoch
    average_loss = total_loss / (len(X_train) / batch_size)
    training_accuracy = correct_predictions / len(X_train)

    val_average_loss = val_total_loss / (len(X_val) / batch_size)
    val_accuracy = val_correct_predictions / len(X_val)

    print(f'Epoch {epoch + 1}/{epochs}, Training Loss: {average_loss}, Training Accuracy: {training_accuracy}, Validation Loss: {val_average_loss}, Validation Accuracy: {val_accuracy}\n')

# Evaluate the model on the test set
hidden_layer_1_input_test = np.dot(X_test_fin, weights_input_hidden_1) + biases_hidden_1
hidden_layer_1_output_test = relu(hidden_layer_1_input_test)

hidden_layer_2_input_test = np.dot(hidden_layer_1_output_test, weights_hidden_1_hidden_2) + biases_hidden_2
hidden_layer_2_output_test = relu(hidden_layer_2_input_test)

hidden_layer_3_input_test = np.dot(hidden_layer_2_output_test, weights_hidden_2_hidden_3) + biases_hidden_3
hidden_layer_3_output_test = relu(hidden_layer_3_input_test)

hidden_layer_4_input_test = np.dot(hidden_layer_3_output_test, weights_hidden_3_hidden_4) + biases_hidden_4
hidden_layer_4_output_test = relu(hidden_layer_4_input_test)

output_layer_input_test = np.dot(hidden_layer_4_output_test, weights_hidden_4_output) + biases_output
output_layer_output_test = softmax(output_layer_input_test, X_test_fin[:,-4:])

test_loss = categorical_crossentropy(y_test_cat, output_layer_output_test)

# Calculate and print test accuracy
test_predictions = np.argmax(output_layer_output_test, axis=1)
test_true_labels = np.argmax(y_test_cat, axis=1)
test_accuracy = np.sum(test_predictions == test_true_labels) / len(X_test_fin)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')