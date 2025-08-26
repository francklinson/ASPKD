import torch.nn as nn
import torch


def init_layer(layer, nonlinearity='leaky_relu'):
    """
    初始化网络层
    """
    nn.init.kaiming_uniform_(layer.weight, nonlinearity=nonlinearity)
    if hasattr(layer, 'bias'):
        if layer.bias is not None:
            layer.bias.data.fill_(0.)


def init_bn(bn):
    """
    初始化批归一化层
    """
    bn.bias.data.fill_(0.)
    bn.running_mean.data.fill_(0.)
    bn.weight.data.fill_(1.)
    bn.running_var.data.fill_(1.)


def init_rnnLayers(rLayer):
    """
    初始化RNN层
    """
    for param in rLayer.parameters():
        if len(param.shape) >= 2:
            torch.nn.init.orthogonal_(param.data)
        else:
            torch.nn.init.normal_(param.data)


class Encoder(nn.Module):
    """

    编码器的主要作用是将输入数据压缩到一个低维的潜在空间（latent space）。
    ### 实现原理

    1. **初始化**：
       - `__init__` 方法中定义了编码器的各个层，包括线性变换层（`nn.Linear`）和激活函数层（`nn.ReLU`）。
       - 使用 `nn.init.xavier_uniform_` 初始化权重，以保持输入和输出的方差相同，有助于网络训练的稳定性。

    2. **前向传播**：
       - `forward` 方法定义了数据在前向传播中的处理流程。
       - 数据通过一系列的线性变换和 ReLU 激活函数。
       - 最后，通过两个线性层分别计算潜在空间的均值和对数方差（log variance），并生成潜在向量。

    ### 用途

    - 在变分自编码器（VAE）中，编码器将输入数据映射到潜在空间，潜在空间中的数据通常具有低维、连续和可解释的特性。
    - 在生成模型中，编码器将输入数据编码成潜在向量，然后通过解码器生成与输入数据相似的新数据。

    ### 注意事项

    - **权重初始化**：权重初始化使用 Xavier 均匀分布，有助于保持网络层的输入和输出方差一致，有助于训练的稳定性。
    - **潜在向量生成**：潜在向量是通过从潜在分布中采样得到的，采样过程使用了标准正态分布和潜在的对数方差。
    - **激活函数**：使用 ReLU 激活函数，因为它可以增加网络的非线性表达能力，同时避免梯度消失问题。

    """
    def __init__(self, input_size: int, hidden1: int, hidden2: int, hidden3: int, hidden4: int, latent_length: int):
        super(Encoder, self).__init__()
        # 定义属性
        self.input_size = input_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.latent_length = latent_length

        # 设定网络
        self.input_to_hidden1 = nn.Linear(self.input_size, self.hidden1)
        self.hidden1_to_hidden2 = nn.Linear(self.hidden1, self.hidden2)
        self.hidden2_to_hidden3 = nn.Linear(self.hidden2, self.hidden3)
        self.hidden3_to_hidden4 = nn.Linear(self.hidden3, self.hidden4)
        # self.hidden4_to_latent = nn.Linear(self.hidden4, self.latent_length)
        self.hidden4_to_mean = nn.Linear(self.hidden4, self.latent_length)
        self.hidden4_to_logvar = nn.Linear(self.hidden4, self.latent_length)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

        nn.init.xavier_uniform_(self.hidden4_to_mean.weight)  # 为了通过网络层时，输入和输出的方差相同 服从均匀分布
        nn.init.xavier_uniform_(self.hidden4_to_logvar.weight)  # 为了通过网络层时，输入和输出的方差相同

        # self.init_weights()

    def init_weights(self):
        init_layer(self.conv1)
        init_layer(self.conv2)
        init_layer(self.conv3)
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)
        init_bn(self.bn5)
        init_bn(self.bn6)
        init_rnnLayers(self.rnnLayer)

    def forward(self, x):
        # img_seq_in = x.view(-1, x.size(0), x.size(1))  # [seq,bach,num_cha]=[154,16,8]
        # 线性变换
        hidden1 = self.ReLU(self.input_to_hidden1(x))
        hidden2 = self.ReLU(self.hidden1_to_hidden2(hidden1))
        hidden3 = self.ReLU(self.hidden2_to_hidden3(hidden2))
        hidden4 = self.ReLU(self.hidden3_to_hidden4(hidden3))
        self.latent_mean = self.hidden4_to_mean(hidden4)
        self.latent_logvar = self.hidden4_to_logvar(hidden4)
        std = torch.exp(0.5 * self.latent_logvar)
        eps = torch.randn_like(std)  # 定义一个和std一样大小的服从标准正态分布的张量
        latent = torch.mul(eps, std) + self.latent_mean  # 标准正太分布乘以标准差后加上均值 latent.shape(batch,latent_length)

        return latent, self.latent_mean, self.latent_logvar  # x.shape(sqe,batch,input)


class Decoder(nn.Module):
    """
    解码器
    将潜在空间（latent space）的表示解码为原始数据空间。
    """
    def __init__(self, output_size: int, hidden1: int, hidden2: int, hidden3: int, hidden4: int, latent_length: int):
        super(Decoder, self).__init__()

        # 定义属性
        self.output_size = output_size
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.hidden4 = hidden4
        self.latent_length = latent_length

        # 设定网络
        self.latent_to_hidden4 = nn.Linear(self.latent_length, self.hidden4)
        self.hidden4_to_hidden3 = nn.Linear(self.hidden4, self.hidden3)
        self.hidden3_to_hidden2 = nn.Linear(self.hidden3, self.hidden2)
        self.hidden2_to_hidden1 = nn.Linear(self.hidden2, self.hidden1)
        self.hidden1_to_output = nn.Linear(self.hidden1, self.output_size)
        self.ReLU = nn.ReLU()
        self.Sigmoid = nn.Sigmoid()

        # self.init_weights()

    def init_weights(self):
        init_layer(self.deconv1)
        init_layer(self.deconv2)
        init_layer(self.deconv3)
        init_layer(self.fc1)
        init_layer(self.fc2)
        init_layer(self.fc3)
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_bn(self.bn3)
        init_bn(self.bn4)
        init_bn(self.bn5)
        init_bn(self.bn6)
        init_rnnLayers(self.rnnLayer)

    def forward(self, latent):
        # 反RNN+线性变换
        hidden4 = self.ReLU(self.latent_to_hidden4(latent))
        hidden3 = self.ReLU(self.hidden4_to_hidden3(hidden4))
        hidden2 = self.ReLU(self.hidden3_to_hidden2(hidden3))
        hidden1 = self.ReLU(self.hidden2_to_hidden1(hidden2))
        output = self.hidden1_to_output(hidden1)
        return output


class Autoencoder(nn.Module):
    """
    自动编码器
    """
    def __init__(self, input_size: int, hidden1: int, hidden2: int, hidden3: int, hidden4: int, latent_length: int):
        """
        :param input_size: 输入数据的维度。
        :param hidden1, hidden2, hidden3, hidden4: 编码器和解码器中各层的隐藏单元数量。
        :param latent_length: 潜在空间的维度。
        :param self.encoder: 定义编码器，将输入数据压缩到潜在空间。
        :param self.decoder: 定义解码器，将潜在空间的数据重建回原始数据。
        """
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_size, hidden1, hidden2, hidden3, hidden4, latent_length)
        self.decoder = Decoder(input_size, hidden1, hidden2, hidden3, hidden4, latent_length)

    def forward(self, x):
        """
        :param x: 输入数据
        :return: x_recon: 重构后的数据，latent: 潜在空间表示，latent_mean: 潜在空间均值，latent_logvar: 潜在空间对数方差

        """
        latent, latent_mean, latent_logvar = self.encoder(x)
        x_recon = self.decoder(latent)
        return x_recon, latent, latent_mean, latent_logvar
