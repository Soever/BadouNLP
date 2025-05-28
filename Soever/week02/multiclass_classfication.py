import torch
import torch.nn as nn
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch.optim as optim
class myNN(nn.Module):
    def __init__(self,input_dim,hidden_dim,output_dim):
        super(myNN, self).__init__()
        self.fc1 = nn.Linear(in_features=input_dim,out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim,out_features=output_dim)
        self.sigmoid =torch.sigmoid
        self.relu = nn.ReLU()
        
    def forward(self,x):
        x = self.fc1(x)
        x = self.relu(x)
        # x = self.sigmoid(x)
        x = self.fc2(x)
        return x

def getSample(sample_num = 10, dim = 5):
    np.random.seed(42)
    X = np.random.rand(sample_num,dim)
    index = np.argmax(X,axis=1)
    labels = np.zeros_like(X)
    labels[np.arange(sample_num),index] = 1  
    return torch.tensor(X,dtype=torch.float),torch.tensor(labels,dtype=torch.float)

def train(X_train,y_train,model,epochs=100,batch_size=64):
    criterion = nn.CrossEntropyLoss()  
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    train_data = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        accuracy = test(model,X_train,y_train)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}, Accuracy on trainset: {accuracy:.4f}")
    return model

def test(model,X_test,y_test):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test)
        _, predictions = torch.max(outputs, dim=1)  # 获取预测的类别索引
        y_test_labels = torch.argmax(y_test, dim=1)
        accuracy = accuracy_score(y_test_labels, predictions.numpy())
    return accuracy
    
    
if __name__ == "__main__":
    sample_num = 5000
    dim = 5
    epochs=50
    batch_size=64
    X,labels = getSample(sample_num=sample_num,dim=dim)
    X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.3, random_state=42)
    model = myNN(input_dim=dim,hidden_dim=32,output_dim=dim)
    model = train(X_train,y_train,model,epochs=epochs,batch_size=batch_size)
    test_accuracy = test(model,X_test,y_test)
    print(f"测试集准确度: {test_accuracy:.4f}")