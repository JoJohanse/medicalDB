import numpy as np
import torch
from MedicineRecommend import MedicineRecommend
from torch.utils.data import DataLoader, TensorDataset, random_split

# 设置使用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载数据
'''
patientDrug.npy (形状(3604, 293)，3604是病人数，293是所有药品数，记录每个病人使用的药品，若使用则为1，否则为0，维度为药品的种类)
patient_vectors.npy (形状(3604, 768)，是每个病人的'出院诊断'属性的BERT编码向量)
drug_vectors.npy (形状(293, 768)，是所有药品(包括所有病人都没用过的)的治疗范围属性的BERT编码)
'''
patient_drug = np.load('patientDrug.npy')
patient_vectors = np.load('patient_vectors.npy')
drug_vectors = np.load('drug_vectors.npy')
# 将patientDrug.npy与patient_vectors.npy根据病人的数量拼接在一起
# 拼接后的形状为(3604, 1061)，3604是病人数，前768位是'出院诊断'属性的BERT编码向量，后293位是该病人所用药品数组
patients = np.concatenate((patient_vectors, patient_drug), axis=1)
# 将3604个病人的数据分离
# 将数据集分为两部分
X = patients[:, :768]  # BERT编码
y = patients[:, 768:]  # 药品01编码

# 转换为Tensor
X = torch.from_numpy(X).float().to(device)
y = torch.from_numpy(y).float().to(device)

# 创建数据集
dataset = TensorDataset(X, y)
# 划分训练集和测试集
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
Train_dataset, Test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
X_train, y_train = Train_dataset[:]
X_test, y_test = Test_dataset[:]
epoches = 100
batch_size = 10

# 编译模型
model = MedicineRecommend(768, 500, 360,293)
model.compile_model()
model.to(device)

# # 训练模型
# model.train_model(X_train, y_train, epoches, batch_size)
# # 保存模型
# model.save('save/model.pth')

# 加载模型
def load(self, path):
    self.model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
load(model, 'save/model.pth')

# 预测
rusults = model.predict_model(X_test)
print("病人",1,"的预测结果为：")
print(rusults[1])
print("test",1,"的真实结果为：")
print(y_test[1])
# 计算准确率
correct = 0
total = 0
for i in range(10):
    for j in range(len(rusults[i])):
        if rusults[i][j] == y_test[i][j]:
            correct += 1
        total += 1
print('Accuracy: {:.2f}%'.format(100 * correct / total))