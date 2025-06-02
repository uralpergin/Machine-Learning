import torch
import numpy as np
#import torch.nn as nn
import pickle
import matplotlib.pyplot as plt

#sof = nn.Softmax(dim=1)
# this function returns the neural network output for a given dataset and set of parameters
def forward_pass(w1, b1, w2, b2, input_data):
    """
    The network consists of 3 inputs, 16 hidden units, and 3 output units
    The activation function of the hidden layer is sigmoid.
    The output layer should apply the softmax function to obtain posterior probability distribution. And the function should return this distribution
    Here you are expected to perform all the required operations for a forward pass over the network with the given dataset
    """
    dot_product_i = torch.matmul(input_data, w1) # 600x3 * 3x16 = 600x16
    output_i = dot_product_i + b1 # 600x16 + 1x16 broadcasted
    
    hidden_layer_output = torch.sigmoid(output_i)
    dot_product_h = torch.matmul(hidden_layer_output, w2) # 600x16 * 16x3
    output_h = dot_product_h + b2 # 600x3 + 1x3 broadcasted

    exp_tensor = torch.exp(output_h)
    row_sum = torch.sum(exp_tensor, dim=1, keepdim=True)
    softmax_result = exp_tensor / row_sum
    

    return softmax_result # 600x3


# we load all training, validation, and test datasets for the classification task
train_dataset, train_label = pickle.load(open("data/part2_classification_train.data", "rb"))
validation_dataset, validation_label = pickle.load(open("data/part2_classification_validation.data", "rb"))
test_dataset, test_label = pickle.load(open("data/part2_classification_test.data", "rb"))

# when you inspect the training dataset, you are going to see that the class instances are sequential (e.g [1,1,1,1 ... 2,2,2,2,2 ... 3,3,3,3])
# we shuffle the training dataset by preserving instance-label relationship
indices = list(range(len(train_dataset)))
np.random.shuffle(indices)
train_dataset = np.array([train_dataset[i] for i in indices], dtype=np.float32)
train_label = np.array([train_label[i] for i in indices], dtype=np.float32)

# In order to be able to work with Pytorch, all datasets (and labels/ground truth) should be converted into a tensor
# since the datasets are already available as numpy arrays, we simply create tensors from them via torch.from_numpy()
train_dataset = torch.from_numpy(train_dataset)
train_label = torch.from_numpy(train_label)

validation_dataset = torch.from_numpy(validation_dataset)
validation_label = torch.from_numpy(validation_label)

test_dataset = torch.from_numpy(test_dataset)
test_label = torch.from_numpy(test_label)

# You are expected to create and initialize the parameters of the network
# Please do not forget to specify requires_grad=True for all parameters since they need to be trainable.

# w1 defines the parameters between the input layer and the hidden layer
w1 = torch.from_numpy(np.random.normal(0, 1, size=(3,16)).astype(np.float32).reshape((3, 16))).requires_grad_(True)
# Here you are expected to initialize w1 via the Normal distribution (mean=0, std=1).
...
# b defines the bias parameters for the hidden layer
b1 = torch.from_numpy(np.random.normal(0, 1, size=(1,16) ).astype(np.float32).reshape((1,16))).requires_grad_(True)
# Here you are expected to initialize b1 via the Normal distribution (mean=0, std=1).
...
# w2 defines the parameters between the hidden layer and the output layer
w2 = torch.from_numpy(np.random.normal(0, 1, size=(16,3)).astype(np.float32).reshape((16, 3))).requires_grad_(True)
# Here you are expected to initialize w2 via the Normal distribution (mean=0, std=1).
...
# and finally, b2 defines the bias parameters for the output layer
b2 = torch.from_numpy(np.random.normal(0, 1, size=(1,3) ).astype(np.float32).reshape((1,3))).requires_grad_(True)
# Here you are expected to initialize b2 via the Normal distribution (mean=0, std=1).
...



# you are expected to use the stochastic gradient descent optimizer
# w1, b1, w2 and b2 are the trainable parameters of the neural network
optimizer = torch.optim.SGD([w1, b1, w2, b2], lr=0.001)
#print("train size %d - validation size %d - test size %d" % (train_label.size(dim=0), validation_label.size(dim=0) , test_label.size(dim = 0)))
# These arrays will store the loss values incurred at every training iteration
iteration_array = []
train_loss_array = []
validation_loss_array = []

# We are going to perform the backpropagation algorithm 'ITERATION' times over the training dataset
# After each pass, we are calculating the average/mean cross entropy loss over the validation dataset along with accuracy scores on both datasets.
ITERATION = 15000
for iteration in range(1, ITERATION+1):
    iteration_array.append(iteration)

    
    # we need to zero all the stored gradient values calculated from the previous backpropagation step.
    optimizer.zero_grad()
    # Using the forward_pass function, we are performing a forward pass over the network with the training data
    train_predictions = forward_pass(w1, b1, w2, b2, train_dataset)

    # here you are expected to calculate the MEAN cross-entropy loss with respect to the network predictions and the training label
    train_mean_crossentropy_loss = -torch.sum(train_label * torch.log(train_predictions)) / train_label.size(dim=0)
    
    
    train_loss_array.append(train_mean_crossentropy_loss.item())

    # We initiate the gradient calculation procedure to get gradient values with respect to the calculated loss 
    train_mean_crossentropy_loss.backward()
    # After the gradient calculation, we update the neural network parameters with the calculated gradients.
    optimizer.step()

    # after each epoch on the training data we are calculating the loss and accuracy scores on the validation dataset
    # with torch.no_grad() disables gradient operations, since during testing the validation dataset, we don't need to perform any gradient operations
    with torch.no_grad():
        # Here you are expected to calculate the accuracy score on the training dataset by using the training labels.
        #print("predictions : ", train_predictions)
        train_pred = torch.argmax(train_predictions, dim=1)
        train_lab = torch.argmax(train_label, dim=1)
        train_comparison = torch.eq(train_pred, train_lab)
        #print("eq num : ",elementwise_comparison)
        train_equal_elements = torch.sum(train_comparison).item()
        #print("pred: ", train_pred)
        #print("label: ", train_lab)
        train_accuracy = (train_equal_elements / train_label.size(dim=0) ) * 100
        
        validation_predictions = forward_pass(w1, b1, w2, b2, validation_dataset)
        
        # Here you are expected to calculate the average/mean cross entropy loss for the validation datasets by using the validation dataset labels.
        validation_mean_crossentropy_loss = -torch.sum(validation_label * torch.log(validation_predictions)) / validation_label.size(dim=0)

        validation_loss_array.append(validation_mean_crossentropy_loss.item())

        # Similarly, here, you are expected to calculate the accuracy score on the validation dataset
        valid_pred = torch.argmax(validation_predictions , dim=1)
        valid_lab = torch.argmax(validation_label , dim=1)
        valid_comparison = torch.eq(valid_pred, valid_lab)

        valid_equal_elements = torch.sum(valid_comparison).item()
        validation_accuracy = (valid_equal_elements / validation_label.size(dim=0) ) * 100
    
    print("Iteration : %d - Train Loss %.4f - Train Accuracy : %.2f - Validation Loss : %.4f Validation Accuracy : %.2f" % (iteration+1, train_mean_crossentropy_loss.item(), train_accuracy, validation_mean_crossentropy_loss.item(), validation_accuracy))


# after completing the training, we calculate our network's accuracy score on the test dataset...
# Again, here we don't need to perform any gradient-related operations, so we are using torch.no_grad() function.
with torch.no_grad():
    test_predictions = forward_pass(w1, b1, w2, b2, test_dataset)
    # Here you are expected to calculate the network accuracy score on the test dataset...
    test_pred = torch.argmax(test_predictions , dim=1)
    test_lab = torch.argmax(test_label , dim=1)
    test_comparison = torch.eq(test_pred, test_lab)

    test_equal_elements = torch.sum(test_comparison).item()
    test_accuracy = (test_equal_elements / test_label.size(dim = 0) ) * 100
    print("Test accuracy : %.2f" % (test_accuracy))

# We plot the loss versus iteration graph for both datasets (training and validation)
plt.plot(iteration_array, train_loss_array, label="Train Loss")
plt.plot(iteration_array, validation_loss_array, label="Validation Loss")
plt.legend()
plt.show()





