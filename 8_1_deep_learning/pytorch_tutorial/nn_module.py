import torch
import torch.nn as nn
import torch.optim as optim

seq_model = nn.Sequential(
    nn.Linear(1, 11),
    nn.Tanh(),
    nn.Linear(11, 1))
print(seq_model)

from collections import OrderedDict

namedseq_model = nn.Sequential(OrderedDict([
    ('hidden_linear', nn.Linear(1, 12)),
    ('hidden_activation', nn.Tanh()),
    ('output_linear', nn.Linear(12, 1))
]))
print(namedseq_model)


class SubclassModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_linear = nn.Linear(1, 13)
        self.hidden_activation = nn.Tanh()
        self.output_linear = nn.Linear(13, 1)

    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = self.hidden_activation(hidden_t)
        output_t = self.output_linear(activated_t)
        return output_t


subclass_model = SubclassModel()
print(subclass_model)

for type_str, model in [('seq', seq_model), ('namedseq', namedseq_model), ('subclass', subclass_model)]:
    print(type_str)
    for name_str, param in model.named_parameters():
        print("{:21} {:19} {}".format(name_str, str(param.shape), param.numel()))
    print()


"""
PyTorch的每个nn模块都有相应的函数。“函数”一词是指“没有内部状态”或“其输出值完全由输入的参数决定”。
实际上，torch.nn.functional提供了许多与nn模块对应的函数，只是所有模型参数（parameter）都作为了参数（argument）移到了函数调用中。
例如，与nn.Linear对应的是nn.functional.linear，它是一个具有参数(input, weight, bias=None)的函数，即模型的权重和偏差是该函数的参数。
"""


class SubclassFunctionalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_linear = nn.Linear(1, 14)
        # 去掉了nn.Tanh()
        self.output_linear = nn.Linear(14, 1)

    def forward(self, input):
        hidden_t = self.hidden_linear(input)
        activated_t = torch.tanh(hidden_t)  # nn.Tanh对应的函数
        output_t = self.output_linear(activated_t)
        return output_t


func_model = SubclassFunctionalModel()
print(func_model)

"""
围绕线性变换的激活函数使神经网络能够逼近高度非线性函数，同时使它们足够简单以容便易优化。
要想识别过拟合，必须将训练集与验证集分开。
没有解决过拟合的诀窍，但是获取更多数据（或数据具有更多可变性）并采用更简单的模型是不错的尝试。
任何从事数据科学的人都应该一直在绘制数据。
"""