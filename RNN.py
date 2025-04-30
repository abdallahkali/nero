import numpy as np
import math
words = ["good", "bad", "great", "awful"]
labels = [1, 0, 1, 0]

def word_to_vector(word, max_len):
    vector = [ord(char) for char in word]
    while len(vector) < max_len:
        vector.append(0)
    return vector

max_len = max(len(w) for w in words)

X = [word_to_vector(word, max_len) for word in words]
y = np.array(labels).reshape(-1, 1)

input_size = max_len  
hidden_size = 16
output_size = 1
learning_rate = 0.01
epochs = 100

np.random.seed(42)  
Wxh = np.random.randn(hidden_size, input_size) * 0.01  
Whh = np.random.randn(hidden_size, hidden_size) * 0.01  
Why = np.random.randn(output_size, hidden_size) * 0.01  
bh = np.zeros((hidden_size, 1))  
by = np.zeros((output_size, 1))  

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

print("Starting training...")
for epoch in range(epochs):
    total_loss = 0
    
    for i in range(len(X)):
        
        x = np.array(X[i]).reshape(-1, 1)  
        target = y[i]
        
        h_prev = np.zeros((hidden_size, 1))  
        
        h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h_prev) + bh)
        y_pred = sigmoid(np.dot(Why, h) + by)
        
        loss = - (target * np.log(y_pred + 1e-10) + (1 - target) * np.log(1 - y_pred + 1e-10))
        total_loss += loss.item()  
        
        dy = y_pred - target
        dWhy = np.dot(dy, h.T)
        dby = dy
        
        dh = np.dot(Why.T, dy) * (1 - h * h)  
        
        dWxh = np.dot(dh, x.T)
        dWhh = np.dot(dh, h_prev.T)
        dbh = dh
        
        Wxh -= learning_rate * dWxh
        Whh -= learning_rate * dWhh
        Why -= learning_rate * dWhy
        bh -= learning_rate * dbh
        by -= learning_rate * dby
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {total_loss/len(X):.4f}")

print("\nTraining completed!\n")

def predict(word):
    
    x = np.array(word_to_vector(word, max_len)).reshape(-1, 1)
    h_prev = np.zeros((hidden_size, 1))
    
    h = np.tanh(np.dot(Wxh, x) + np.dot(Whh, h_prev) + bh)
    y_pred = sigmoid(np.dot(Why, h) + by)
    return "Positive" if y_pred > 0.5 else "Negative"
test_words = ["good", "bad", "excellent", "terrible", "nice", "horrible"]
print("Testing the model:")
for word in test_words:
    if len(word) > max_len:
        print(f"Word '{word}' is too long (max length is {max_len})")
    else:
        print(f"Word '{word}': Prediction: {predict(word)}")