import torch
from torchvision.datasets import FashionMNIST
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor
from Models import LeNet
from torch import nn

BATCH_SIZE = 64
lr = 1e-3
EPOPCHS = 50

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dataset = FashionMNIST(root='data', train=True, transform=ToTensor(), download=True)
test_dataset = FashionMNIST(root='data', train=False, transform=ToTensor(), download=True)


train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

model = LeNet(10)

print(model)

def train_loop(model, dataloader, loss_fn, optimizer):
    model.train()
    size = len(dataloader.dataset)
    
    for batch, (X, y) in enumerate(dataloader):
        pred = model(X)
        
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        
        if batch % 100 == 0:
            loss, current = loss.item(), batch * BATCH_SIZE + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            
def test_loop(model, dataloader, loss_fn):
    model.eval()
    
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            
            test_loss+= loss_fn(pred, y).item()
            correct+= (pred.argmax(1) == y).type(torch.float).sum().item()
    
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    
    
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

for epoch in range(EPOPCHS):
    print(f"Epoch {epoch+1}\n-------------------------------")
    train_loop(model, train_dataloader, loss_fn, optimizer)
    test_loop(model, test_dataloader, loss_fn)
        