# 导入的torch是神经网络的一个模块
import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# 条件判断cuda
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")


# 自己定义了一个MLP，并继承了nn.Module模块——构架自己的网络
class MLP(nn.Module):
    def __init__(self, input_size):
        # super(NeuralNetwork, self).__init__()
        super().__init__()      # 继承
        self.flatten = nn.Flatten()   # 属性
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.ReLU(),      # 激活模型
            nn.Linear(512, 512),    # 线性函数
            nn.ReLU(),
            nn.Linear(512, 2),
        )

    def forward(self, x):
        x = self.flatten(x)    # 调用函数，展平
        logits = self.linear_relu_stack(x)
        return logits

model = MLP(input_size=4096).to(device)
print(model)    # 得到一个神经网络模型


# 保存与加载模型
torch.save(model.state_dict(), 'model_weights.pth')
print(model.state_dict())
torch.save(model, 'model.pth')
model = torch.load('model.pth')

# 构建一个随机三维张量 通道数只有一个，横向和纵向都有64个值，显然是一个灰度图像
X = torch.rand(1, 64, 64, device=device)
logits = model(X)


# 用softmax函数网络层，得到了一个logits结果
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")   # 输出概率值

input_image = torch.rand(3,28,28)
print(input_image.size())

# 展平
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)   # 隐藏层
print(hidden1.size())

# 得到极激活函数，和隐藏层
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")


seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3,28,28)
logits = seq_modules(input_image)

softmax = nn.Softmax(dim=1)   # 把logits结果，用softmax函数映射到一个概率向量中，来判断最后的类别
pred_probab = softmax(logits)


print(f"Model structure: {model}\n\n")
for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")



    
    
    
    
    
    