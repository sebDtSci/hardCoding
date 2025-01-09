import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.weight_input_hidden = np.random.rand(input_size, hidden_size)
        self.weight_output_hidden = np.random.rand(hidden_size, output_size)
        self.bias_hidden = np.zeros((1, hidden_size))
        self.bias_output = np.zeros((1, output_size))

    def tan(self, x):
        return (2/(1+np.exp(-2*x)))-1
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x))
        return exp_x / exp_x.sum(axis=1, keepdims=True)

    def forward(self, X):
        self.hidden_input = np.dot(X, self.weight_input_hidden) + self.bias_hidden
        # self.hidden_output = self.tan(self.hidden_input)
        self.hidden_output = self.sigmoid(self.hidden_input)
        self.final_intput = np.dot(self.hidden_output, self.weight_output_hidden) + self.bias_output
        self.final_output = self.softmax(self.final_intput)
        return self.final_output
    
    def backward(self, X, y, output, lr):
        output_error = output - y
        hidden_error = np.dot(output_error, self.weight_output_hidden.T) * self.hidden_output * (1 - self.hidden_output)

        self.weight_output_hidden -= lr * np.dot(self.hidden_output.T, output_error)
        self.bias_output -= lr * np.sum(output_error, axis=0, keepdims=True)
        self.weight_input_hidden -= lr * np.dot(X.T, hidden_error)
        self.bias_hidden -= lr * np.sum(hidden_error, axis=0, keepdims=True)

    def train(self, X, y, epochs, learning_rate):
        epsilon = 1e-8
        mse_hist = []
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output, learning_rate)
            if (epoch+1) % 100 == 0:
                # loss = -np.sum(y * np.log(output)) / X.shape[0]
                # pour éviter des problèmes numériques (logarithme de 0), ajout d'une petite valeur epsilon :
                loss = -np.sum(y * np.log(output + epsilon)) / X.shape[0]
                mse = np.square(np.subtract(y,output)).mean()
                mse_hist.append(mse)
                print(f'Epoch {epoch+1}, Loss: {loss:.4f}, MSE: {mse:.4f}')
        return {"mse" : mse_hist}
    
    def predict(self, X):
        output = self.forward(X)
        return np.argmax(output, axis=1)