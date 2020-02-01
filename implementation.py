import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import matplotlib.cm
import numpy as np
import cv2
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score
import warnings

warnings.filterwarnings("ignore")
print('Warnings are turned off')

# Hyper parameters
n_epochs = 10
batch_size_train = 32
batch_size_test = 1000
learning_rate = 0.012
# Log parameters and seed
log_interval = 10
random_seed = 1
torch.manual_seed(random_seed)

# Load MNIST data. One for training the other for inference and Re-identification
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()])),
    batch_size=batch_size_train, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()])),
    batch_size=batch_size_test, shuffle=True)

full_train_data = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=False, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()])),
    batch_size=60000)

full_test_data = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/files/', train=True, download=True, transform=torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()])),
    batch_size=1)


# Network architecture
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(32, 2)
        self.fc2 = nn.Linear(2, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=7))
        x = x.view(-1, 32)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.log_softmax(x)

    def features(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv2(x), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=7))
        x = x.view(-1, 32)
        x = self.fc1(x)
        return x


# If GPU available use the line .cuda(), otherwise remove.
network = Net()
network.cuda() # Transfer to GPU
optimizer = optim.SGD(network.parameters(), lr=learning_rate)

# Empty lists to append losses and accuracies to
batch_losses = []
batch_accuracies = []
train_losses = []
test_losses = []
total_accuracies = []
re_id_accuracies = []
epoch_vectors = []

model = Net()


# Re identification accuracy
def re_id():
    model.load_state_dict(torch.load('Model/state.pt'))
    model.eval()

    total_score = 0
    for data, target in full_train_data:
        train_gallery = model.features(data)
        train_gallery = train_gallery.detach().numpy()
        # Closest Euclidean distance -> KNN
        knn_gallery = NearestNeighbors(20).fit(train_gallery)

        # Test the KNN fit
        q = 0
        for data2, target2 in full_test_data:
            data2 = model.features(data2)
            data2 = data2.detach().numpy()
            predictions = knn_gallery.kneighbors(data2, return_distance=False)
            prediction_tensor = torch.LongTensor(predictions[0]).squeeze()
            label_hat = torch.index_select(target, 0, prediction_tensor)
            score = accuracy_score(np.full(20, target2.detach().numpy()), label_hat)
            total_score += score
    total_score /= 1000
    # Return accuracy and the k nearest neighbor fit
    return total_score, knn_gallery


# Train loop
def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        network.train()
        optimizer.zero_grad()
        data = data.cuda() # Transfer to GPU
        target = target.cuda() # Transfer to GPU
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        batch_accuracy = train_loss(data, target)
        batch_accuracy = batch_accuracy.item()
        train_losses.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tBatch accuracy: {}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item(), batch_accuracy))
        torch.save(network.state_dict(), 'Model/state.pt')

# Calculates loss and accuracy per batch
def train_loss(data, target):
    network.eval()
    batch_loss = 0
    correct = 0
    data = data.cuda() # Transfer to GPU
    target = target.cuda() # Transfer to GPU
    with torch.no_grad():
        output = network(data)
        batch_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
    batch_loss /= batch_size_train
    batch_losses.append(batch_loss)
    batch_accuracy = 100. * correct / len(target)
    batch_accuracies.append(batch_accuracy)

    return batch_accuracy


# Test all data
def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data = data.cuda() # Transfer to GPU
            target = target.cuda() # Transfer to GPU
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return 100. * correct / len(test_loader.dataset)


# Combine digits to track as a tensor
def track():
    for batch_idx, (data, target) in enumerate(full_train_data):
        # Select the 20 images per digit to track in embedding space
        first_twenty = torch.tensor(range(20))
        numbers = []
        for x in range(10):
            index = (target == x).nonzero().reshape(-1)
            cache = torch.index_select(data, 0, index)
            cache = torch.index_select(cache, 0, first_twenty)
            numbers.append(cache)

    saved = torch.cat((numbers[0], numbers[1], numbers[2], numbers[3], numbers[4], numbers[5], numbers[6], numbers[7],
                       numbers[8], numbers[9]), 0)
    # Return data with 20 images per class
    return saved


# Make initial embedding space, then train and make corresponding embedding space
vectors = 0
subset = track().cuda() # Transfer to GPU
for epoch in range(0, n_epochs + 1):
    if epoch == 0:
        vectors = network.features(subset)
        vectors = vectors.cpu().detach().numpy()
        vectors = np.split(vectors, 10)
        colors = matplotlib.cm.Paired(np.linspace(0, 1, len(vectors)))
        fig, ax = plt.subplots()
        ax.set_title('Embedding space')
        ax.set_xlabel('x1 (node 1)')
        ax.set_ylabel('x2 (node 2)')

        for (points, color, digit) in zip(vectors, colors, range(10)):
            ax.scatter([item[0] for item in points],
                       [item[1] for item in points], color=color, label='digit    {}'.format(digit))
            ax.grid(True)
        ax.legend()
        continue

    train(epoch)
    vectors = network.features(subset)
    vectors = vectors.cpu().detach().numpy()
    vectors = np.split(vectors, 10)
    colors = matplotlib.cm.Paired(np.linspace(0, 1, len(vectors)))
    fig, ax = plt.subplots()
    ax.set_title('Embedding space')
    ax.set_xlabel('x1 (node 1)')
    ax.set_ylabel('x2 (node 2)')
    ax.set_xlim([-25, 25])
    ax.set_ylim([-25, 25])

    for (points, color, digit) in zip(vectors, colors, range(10)):
        ax.scatter([item[0] for item in points],
                   [item[1] for item in points], color=color, label='digit    {}'.format(digit))
        ax.grid(True)
    ax.legend()

    epoch_vectors.append(vectors)
    accuracy = test()
    accuracy = accuracy.item()
    re_id_acc, knn = re_id()

    print(f'ReID accuracy: {round(re_id_acc, 2)}')
    total_accuracies.append(accuracy)
    re_id_accuracies.append(re_id_acc)

# Checkpoint file
checkpoint = {'model': Net(), 'state_dict': network.state_dict(), 'optimizer': optimizer.state_dict()}
torch.save(checkpoint, 'Model/checkpoint.pth')

# List to iterate through digits
digits = ['zero', 'one', 'two', 'three',
          'four', 'five', 'six',
          'seven', 'eight', 'nine']

neighbours = []
vecs = []

# Find nearest neighbours of own handwriting
for digit in digits:
    img = cv2.imread('Model/Digits/' + digit + '.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (22, 22))
    img = cv2.bitwise_not(img) # invert colours
    _, img = cv2.threshold(img, 60, 0, cv2.THRESH_TRUNC)
    _, img = cv2.threshold(img, 15, 0, cv2.THRESH_TOZERO)
    img = cv2.copyMakeBorder(img, 3, 3, 3, 3, cv2.BORDER_CONSTANT, 0)
    maxx = np.amax(img)
    img = torch.from_numpy(img).float()
    img = torch.div(img, maxx)
    img = img.view(1, 1, 28, 28)
    value = network(img.cuda()) # Transfer to GPU
    vector = network.features(img.cuda()) # Transfer to GPU
    vector = vector.cpu().detach().numpy()
    vecs.append(vector)
    neighbour = knn.kneighbors(vector, return_distance=False)

    neighbour = torch.from_numpy(neighbour[0][0:5])
    for idx, (data, target) in enumerate(full_train_data):
        datapoints = torch.index_select(data, 0, neighbour)
        chunks = torch.chunk(datapoints, 5, 0)
        images = [x.squeeze().detach().numpy() for x in chunks]
        neighbours.extend(images)

colors = matplotlib.cm.Paired(np.linspace(0, 1, len(vectors)))
fig, ax = plt.subplots()
ax.set_xlim([-25, 25])
ax.set_ylim([-25, 25])
ax.set_title('Embedding space')
ax.set_xlabel('x1 (node 1)')
ax.set_ylabel('x2 (node 2)')
for (points, color, dig) in zip(vectors, colors, range(10)):
    ax.scatter([item[0] for item in points],
               [item[1] for item in points], color=color, label='Digit {}'.format(dig))

for (points, color, dig) in zip(vecs, colors, range(10)):
    ax.scatter([item[0] for item in points],
               [item[1] for item in points], color=color, marker='x')

ax.grid(True)
ax.legend()

# Plot figures
lines = [1875, 3750, 5625, 7500, 9375, 11250, 13125, 15000, 16875] # For vertical epoch lines
batch_loss_plot, axs_1 = plt.subplots(constrained_layout=True, figsize=(10, 4.8))
axs_1.plot(batch_losses)
axs_1.set_title('Loss per batch')

for line in lines:
    plt.axvline(line, color='r', linestyle='--')

axs_1.set_xlabel('Batch number')
axs_1.set_ylabel('Loss')
axs_1.set_xlim([0, 18750])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

batch_acc_plot, axs_2 = plt.subplots(figsize=(10, 4.8))
axs_2.plot(batch_accuracies)
axs_2.set_title('Accuracy per Batch')
axs_2.set_ylim([0, 100])
axs_2.set_xlim([0, 18750])
for line in lines:
    plt.axvline(line, color='r', linestyle='--')

axs_2.set_xlabel('Batch number')
axs_2.set_ylabel('Accuracy (%)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

total_accuracies_plot, axs_3 = plt.subplots()
axs_3.plot(range(1, 11), total_accuracies)
axs_3.set_title('Accuracy per epoch')
axs_3.set_xlabel('Epoch (#)')
axs_3.set_ylabel('Accuracy (%)')
axs_3.set_ylim([30, 100])
axs_3.set_xlim([1, 10])
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)

re_id_accuracies_plot, axs_4 = plt.subplots()
axs_4.plot(range(1, 11), re_id_accuracies)
axs_4.set_title('ReID accuracy per epoch')
axs_4.set_ylim([20, 60])
axs_4.set_xlim([1, 10])
axs_4.set_xlabel('Epoch (#)')
axs_4.set_ylabel('Accuracy (%)')
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)


fig, axes = plt.subplots(5, 10)
k = 0
i = 0
for img in neighbours:
    axes[i, k].imshow(img, cmap='gray')
    axes[i, k].axis('off')
    if i == 0:
        axes[i, k].set_title(f'#{k}')
    i += 1
    if i == 5:
        k += 1
        i = 0

plt.show()




