import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from matplotlib.pyplot import imshow
from torchvision import datasets, transforms

# Feel free to import other packages, if needed.
# As long as they are supported by CSL machines.


def get_data_loader(training = True):
    """
    TODO: implement this function.

    INPUT: 
        An optional boolean argument (default value is True for training dataset)

    RETURNS:
        Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])
    #dataset = torchvision.datasets.FashionMNIST()
    train_set = datasets.FashionMNIST('./data', train=True, download=True,transform=transform)
    test_set = datasets.FashionMNIST('./data', train=False, transform=transform)
    if not training:
        return torch.utils.data.DataLoader(test_set, batch_size=64, shuffle=False)
    return torch.utils.data.DataLoader(train_set, batch_size=64)





def build_model():
    """
    TODO: implement this function.

    INPUT: 
        None

    RETURNS:
        An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),        # TODO: check if 10 channel images?
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    return model



def train_model(model, train_loader, criterion, T):
    """
    TODO: implement this function.

    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train()

    for epoch in range(T):
        total = 0
        correct = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            opt.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(f'Train Epoch: {epoch}    Accuracy: {correct}/{total}({100*correct/total:.2f}%)     Loss: {loss:.3f}')







def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    TODO: implement this function.

    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cropy-entropy 

    RETURNS:
        None
    """
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for data, labels in test_loader:
            outputs = model(data)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    if not show_loss:
        print(f'Accuracy: {100*correct/total:.2f}%')
    else:
        print(f'Average loss: {loss:.4f}\nAccuracy: {100*correct/total:.2f}%')

def predict_label(model, test_images, index):
    """
    TODO: implement this function.

    INPUT: 
        model - the trained model
        test_images   -  test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1


    RETURNS:
        None
    """
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot']

    outputs = model(test_images[index])
    prob = F.softmax(outputs, dim=1)
    _, predicted = torch.max(prob, dim=1)
    limit = 2
    index = 0
    for i in range(3):
        maxVal = prob[0][0].item()
        for i in range(len(class_names)):
            largest = prob[0][i].item()
            if limit > largest > maxVal:
                maxVal = largest
                index = i
        print(f'{class_names[index]}: {maxVal * 100:.2f}%')
        limit = maxVal








if __name__ == '__main__':
    '''
    Feel free to write your own test code here to exaime the correctness of your functions. 
    Note that this part will not be graded.
    '''

