import torch
import torch.nn as nn
import numpy as np
import pickle

class MLP(nn.Module):
    def __init__(self, input_size, num_hidden_layers, hidden_size, output_size, activation):
        super(MLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.layers = nn.ModuleList()
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(input_size, hidden_size))
            if activation == 'relu':
                self.layers.append(nn.ReLU())
            elif activation == 'tanh':
                self.layers.append(nn.Tanh())
            elif activation == 'sigmoid':
                self.layers.append(nn.Sigmoid())
            input_size = hidden_size
        self.output_layer = nn.Linear(input_size, output_size)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.output_layer(x)
    

# we load all the datasets of Part 3
x_train, y_train = pickle.load(open("data/mnist_train.data", "rb"))
x_validation, y_validation = pickle.load(open("data/mnist_validation.data", "rb"))
x_test, y_test = pickle.load(open("data/mnist_test.data", "rb"))

x_train = x_train/255.0
x_train = x_train.astype(np.float32)

x_test = x_test / 255.0
x_test = x_test.astype(np.float32)

x_validation = x_validation/255.0
x_validation = x_validation.astype(np.float32)

# and converting them into Pytorch tensors in order to be able to work with Pytorch
x_train = torch.from_numpy(x_train)
y_train = torch.from_numpy(y_train).to(torch.long)

x_validation = torch.from_numpy(x_validation)
y_validation = torch.from_numpy(y_validation).to(torch.long)

x_test = torch.from_numpy(x_test)
y_test = torch.from_numpy(y_test).to(torch.long)



num_hidden_layers = [1, 2, 3]
hidden_size = [64, 128, 256]
learning_rates = [0.001, 0.01, 0.1]
num_epochs = [10, 20, 30]
activations = ['relu', 'tanh', 'sigmoid']


# Initialize variables to track test accuracy for each set of hyperparameters
best_hyperparameters = None
best_accuracy = 0


soft_max_function = torch.nn.Softmax(dim=1)


    
iteration = 0
confidence_score = 0
# Iterate through hyperparameters
criterion = nn.CrossEntropyLoss()
for layers in num_hidden_layers:
    for size in hidden_size:
        for lr in learning_rates:
            for epochs in num_epochs:
                for activation in activations:
                    model = MLP(784, layers, size, 10, activation)
                    iteration+=1
                    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                    train_accuracy_scores = []
                    for _ in range(10):# Repeat the process 10 times
                        
                        # Training loop
                        for epoch in range(epochs):
                            optimizer.zero_grad()
                            outputs = model(x_train)
                            loss = criterion(outputs, y_train)
                            loss.backward()
                            optimizer.step()

                        with torch.no_grad():# Test the model on the train dataset
                            train_outputs = model(x_train)
                            train_probs = soft_max_function(train_outputs)
                            train_preds = torch.argmax(train_probs, dim=1)
                            
                            train_accuracy = ( (train_preds == y_train).sum().item() / len(y_train) ) * 100
                            train_accuracy_scores.append(train_accuracy)

                    with torch.no_grad():
                        train_accuracy_tensor = torch.FloatTensor(train_accuracy_scores)
                        mean_train_accuracy = torch.mean(train_accuracy_tensor)
                        std_train_accuracy = torch.std(train_accuracy_tensor)
                        confidence_interval_train = 1.96 * (std_train_accuracy / np.sqrt(len(train_accuracy_scores)))

                        print("ITERATION: %d / 243" % iteration)
                        print("layer: %d - size: %d - lr: %3f - epochs: %d - activation: %s" %(layers, size, lr, epochs, activation))
                        print("Mean Train Accuracy: ", mean_train_accuracy.item())
                        print("Confidence Interval Train (95%): ", confidence_interval_train.item())
                        print(" ")# find best hyperparameters
                        if(confidence_interval_train > confidence_score):
                            confidence_score = confidence_interval_train
                            best_hyperparameters = (layers, size, lr, epochs, activation)
                        




combined_data = torch.cat((x_train, x_validation))
combined_labels = torch.cat((y_train, y_validation)) 
model = MLP(784, best_hyperparameters[0], best_hyperparameters[1], 10, best_hyperparameters[4])

optimizer = torch.optim.Adam(model.parameters(), lr=best_hyperparameters[2])

test_accuracy_scores = []
for _ in range(10):
   

    # Training loop
    for epoch in range(best_hyperparameters[3]):
        optimizer.zero_grad()
        outputs = model(combined_data)
        loss = criterion(outputs, combined_labels)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        test_outputs = model(x_test)
        test_probs = nn.Softmax(test_outputs, dim=1)
        test_preds = torch.argmax(test_probs, dim=1)
        test_accuracy = ( (test_preds == y_test).sum().item() / len(y_test) ) * 100
        test_accuracy_scores.append(test_accuracy)

with torch.no_grad():
    test_accuracy_tensor = torch.FloatTensor(test_accuracy_scores)
    mean_test_accuracy = torch.mean(test_accuracy_tensor)
    std_test_accuracy = torch.std(test_accuracy_tensor)
    confidence_interval = 1.96 * (std_test_accuracy / np.sqrt(len(test_accuracy_scores)))
    print("Mean Test Accuracy: ", mean_test_accuracy.item())
    print("Confidence Interval Test (95%): ", confidence_interval.item())        
    print("BEST PARAMETERS:")
    print("Hidden Layer Number: ", best_hyperparameters[0])
    print("Hidden Layer Size: ", best_hyperparameters[1])
    print("Learning Rate: ", best_hyperparameters[2])
    print("Epoch: ", best_hyperparameters[3])
    print("Activation Function: ", best_hyperparameters[4])