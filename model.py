#!/usr/bin/env python
# coding: utf-8

# In[63]:


# Import required libraries
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import shuffle
import pickle

# Step 1: Set up the paths to your image folders
train_path = r"../../archive/skin-disease-datasaet/train_set"
test_path = r"../../archive/skin-disease-datasaet/test_set"

# Check if the paths exist
if not os.path.exists(train_path) or not os.path.exists(test_path):
    raise FileNotFoundError("Please check your dataset paths!")

# Step 2: Function to load and prepare images
def load_images(folder_path, image_size=(64, 64)):
    """Load images from folders and prepare them for training."""
    images = []  # Will store all images
    labels = []  # Will store corresponding labels
    
    # Go through each subfolder (each disease type)
    for disease_name in os.listdir(folder_path):
        disease_folder = os.path.join(folder_path, disease_name)
        if os.path.isdir(disease_folder):
            print(f"Loading images for: {disease_name}")
            
            # Load each image in the disease folder
            for image_name in os.listdir(disease_folder):
                image_path = os.path.join(disease_folder, image_name)
                try:
                    # Open, resize, and convert image to RGB
                    image = Image.open(image_path)
                    image = image.resize(image_size)
                    image = image.convert('RGB')
                    
                    # Convert image to numpy array and add to our lists
                    images.append(np.array(image))
                    labels.append(disease_name)
                except Exception as e:
                    print(f"Couldn't load image {image_name}: {e}")
    
    return np.array(images), np.array(labels)

def transform(images):
    return images.astype('float32') / 255.0
# Step 3: Function to prepare data for training
def prepare_data(images, labels):
    """Convert images and labels into the right format for training."""
    # Normalize pixel values to be between 0 and 1
    X = transform(images)
    
    # Convert text labels to numbers
    label_converter = LabelEncoder()
    numeric_labels = label_converter.fit_transform(labels)
    
    # Convert to one-hot encoding (e.g., 2 -> [0,0,1,0,0])
    num_classes = len(set(numeric_labels))
    y = np.zeros((len(numeric_labels), num_classes))
    y[np.arange(len(numeric_labels)), numeric_labels] = 1
    
    return X, y, label_converter

# Step 4: Neural Network Class
class MultilayerNeuralNetwork:
    def __init__(self, layer_sizes=[],labels={},training=True):
        """
        Initialize neural network with multiple layers.
        layer_sizes: list of numbers representing the size of each layer
        Example: [12288, 512, 256, 128, 7] for 3 hidden layers
        """
        # Initialize weights and biases for each layer
        self.weights = {}
        self.biases = {}
        self.layers = {}
        self.activations = {}
        self.labels=labels
        
        if(training == False):
            with open("parameters.pickle", "rb") as infile:
                parameters = pickle.load(infile)
                self.weights = parameters[0]
                self.biases = parameters[1]
                layer_sizes = self.layer_sizes = parameters[2]
                self.labels = parameters[3]

        self.num_layers = len(layer_sizes) - 1
        
        
        # Create weights and biases for each layer
        if(training):
            for i in range(self.num_layers):
                # He initialization for better training
                self.weights[i] = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2./layer_sizes[i])
                self.biases[i] = np.zeros((1, layer_sizes[i+1]))
            self.layer_sizes = layer_sizes
            
        # Storage for training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
    
    def relu(self, x):
        """Activation function: returns x if x > 0, else 0"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        """Derivative of ReLU function"""
        return (x > 0) * 1
    
    def softmax(self, x):
        """Convert numbers to probabilities that sum to 1"""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
    
    def forward(self, X):
        """Forward pass through all layers"""
        current_input = X
        
        # Pass through each layer except the last
        for i in range(self.num_layers - 1):
            # Compute layer output
            self.layers[i] = np.dot(current_input, self.weights[i]) + self.biases[i]
            # Apply ReLU activation
            self.activations[i] = self.relu(self.layers[i])
            current_input = self.activations[i]
        
        # Last layer with softmax
        i = self.num_layers - 1
        self.layers[i] = np.dot(current_input, self.weights[i]) + self.biases[i]
        self.activations[i] = self.softmax(self.layers[i])
        
        return self.activations[i]
    
    def backward(self, X, y, learning_rate):
        """Backward pass to update weights"""
        batch_size = X.shape[0]
        
        # Initialize gradients
        dweights = {}
        dbiases = {}
        
        # Output layer error
        error = self.activations[self.num_layers-1] - y
        
        # Go through layers backwards
        for i in range(self.num_layers-1, -1, -1):
            # Calculate gradients for weights and biases
            if i == 0:
                dweights[i] = np.dot(X.T, error) / batch_size
            else:
                dweights[i] = np.dot(self.activations[i-1].T, error) / batch_size
            dbiases[i] = np.sum(error, axis=0, keepdims=True) / batch_size
            
            # Calculate error for next layer
            if i > 0:
                error = np.dot(error, self.weights[i].T) * self.relu_derivative(self.layers[i-1])
        
        # Update weights and biases
        for i in range(self.num_layers):
            self.weights[i] -= learning_rate * dweights[i]
            self.biases[i] -= learning_rate * dbiases[i]
    
    def compute_loss(self, y_true, y_pred):
        """Calculate the loss (how wrong the predictions are)"""
        return -np.mean(y_true * np.log(y_pred + 1e-15))
    
    def compute_accuracy(self, y_true, y_pred):
        """Calculate the accuracy (percentage of correct predictions)"""
        predicted_classes = np.argmax(y_pred, axis=1)
        true_classes = np.argmax(y_true, axis=1)
        return np.mean(predicted_classes == true_classes)
    
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=32, learning_rate=0.01):
        """Train the neural network"""
        num_samples = len(X_train)
        
        for epoch in range(epochs):
            # Shuffle the training data
            indices = np.random.permutation(num_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            # Train in batches
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Forward and backward pass
                self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
            
            # Calculate metrics
            train_pred = self.forward(X_train)
            train_loss = self.compute_loss(y_train, train_pred)
            train_acc = self.compute_accuracy(y_train, train_pred)
            
            val_pred = self.forward(X_val)
            val_loss = self.compute_loss(y_val, val_pred)
            val_acc = self.compute_accuracy(y_val, val_pred)
            
            # Store history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            # Print progress
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}/{epochs}")
                print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
                print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")
                print("-" * 50)

        parameters=[self.weights,self.biases,self.layer_sizes,self.labels]
        print("parameters:",parameters)
        with open("parameters.pickle", "wb") as outfile:
            pickle.dump(parameters, outfile)
            
    
    def plot_history(self):
        """Plot training history"""
        plt.figure(figsize=(12, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Training Loss')
        plt.plot(self.history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Training Accuracy')
        plt.plot(self.history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    def transform(self,output):
        return self.labels[np.argmax(output)]


# In[64]:


def train():
    # Step 5: Load and prepare the data
    print("Loading training images...")
    train_images, train_labels = load_images(train_path)
    print("Loading test images...")
    test_images, test_labels = load_images(test_path)

    # Prepare the data
    print("Preparing data for training...")
    X_train, y_train, label_converter = prepare_data(train_images, train_labels)
    X_test, y_test, _ = prepare_data(test_images, test_labels)

    # Flatten the images
    X_train_flat = X_train.reshape(X_train.shape[0], -1)
    X_test_flat = X_test.reshape(X_test.shape[0], -1)

    # Step 6: Create and train the model
    input_size = X_train_flat.shape[1]  # Number of pixels in each image
    output_size = len(set(train_labels))  # Number of disease classes

    # Define the size of each layer
    layer_sizes = [
        input_size,    # Input layer
        512,           # First hidden layer
        256,           # Second hidden layer
        128,           # Third hidden layer
        output_size    # Output layer
    ]
        # Print class labels
    label_index = {}
    print("\nDisease Classes:")
    for i, label in enumerate(label_converter.classes_):
        print(f"{i}: {label}")
        label_index[i] = label

    print("Creating neural network...")
    model = MultilayerNeuralNetwork(layer_sizes,labels=label_index)
    print("Starting training...")
    model.train(
        X_train_flat, y_train,
        X_test_flat, y_test,
        epochs=100,
        batch_size=32,
        learning_rate=0.01
    )

    # Plot the training history
    model.plot_history()

    # Final evaluation
    final_predictions = model.forward(X_test_flat)
    final_accuracy = model.compute_accuracy(y_test, final_predictions)
    print(f"\nFinal Test Accuracy: {final_accuracy * 100:.2f}%")


# In[65]:


def predict(image):
    processed_image = transform(image)
    input = processed_image.reshape(1, -1)
    model = MultilayerNeuralNetwork(training=False)
    return model.transform(model.forward(input))

