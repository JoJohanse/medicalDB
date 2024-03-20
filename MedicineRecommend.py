import torch
from torch import nn
from torch.optim import Adam


class MedicineRecommend(nn.Module):
    def __init__(self, input_dim, hidden_dim1, hidden_dim2, output_dim):
        super(MedicineRecommend, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim1),
            nn.ReLU(),
            nn.Linear(hidden_dim1, hidden_dim2),
            nn.ReLU(),
            nn.Linear(hidden_dim2, output_dim),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

    def compile_model(self):
        self.criterion = nn.BCELoss()
        self.optimizer = Adam(self.parameters())

    def train_model(self, X_train, y_train, epochs, batch_size):
        for epoch in range(epochs):
            for i in range(0, X_train.size(0), batch_size):
                X_batch = X_train[i:i + batch_size]
                y_batch = y_train[i:i + batch_size]
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                # 打印loss
                if i % 100 == 0:
                    print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch + 1, epochs, i + 1, X_train.size(0), loss.item()))
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def evaluate_model(self, X_test, y_test):
        with torch.no_grad():
            outputs = self.model(X_test)
            loss = self.criterion(outputs, y_test)
            # 计算准确率
            for i in range(len(outputs)):
                for j in range(len(outputs[i])):
                    if outputs[i][j] > 0.5:
                        outputs[i][j] = 1
                    else:
                        outputs[i][j] = 0
            correct = 0
            total = 0
            for i in range(len(outputs)):
                for j in range(len(outputs[i])):
                    if outputs[i][j] == y_test[i][j]:
                        correct += 1
                    total += 1
            print('Accuracy: {:.2f}%'.format(100 * correct / total))
            return loss.item()

    def predict_model(self, X):
        with torch.no_grad():
            outputs = self.model(X)
            for i in range(len(outputs)):
                for j in range(len(outputs[i])):
                    if outputs[i][j] > 0.5:
                        outputs[i][j] = 1
                    else:
                        outputs[i][j] = 0
            return outputs

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))