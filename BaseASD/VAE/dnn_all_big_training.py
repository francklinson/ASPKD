"""
这段代码的主要功能是使用变分自编码器（VAE）来计算潜在表示（latent representation），
并使用高斯混合模型（Gaussian Mixture Model, GMM）来评估这些潜在表示。具体来说，这段代码实现了以下步骤：

1. **导入必要的库**：
   - `torch`：PyTorch库，用于深度学习。
   - `torch.nn`：PyTorch神经网络模块。
   - `torch.optim`：PyTorch优化器模块。
   - `torch.utils.data`：PyTorch数据加载模块。
   - `sklearn`：Scikit-learn库，用于机器学习。
   - 自定义模块：`data`、`Autoencoder`、`common`。

2. **设备选择**：
   - 根据是否有可用的GPU，选择使用CPU或GPU进行计算。

3. **参数加载**：
   - 从YAML文件中加载模型和训练参数。

4. **计算潜在表示**：
   - 定义一个函数`calculation_latent`，该函数接受数据类型、ID、训练数据路径和测试数据路径作为输入。
   - 加载训练和测试数据。
   - 初始化变分自编码器（VAE）。
   - 定义损失函数和优化器。
   - 将模型和数据移动到选择的设备上。
   - 创建数据加载器。
   - 进行多轮训练和验证，计算损失和潜在表示。
   - 使用高斯混合模型（GMM）评估潜在表示。
   - 保存训练过程中的损失和潜在表示。
   - 每轮训练结束后保存模型。

5. **主程序**：
   - 遍历不同的ID，调用`calculation_latent`函数进行训练和验证。

### 实现原理

- **变分自编码器（VAE）**：
  - VAE是一种生成模型，通过编码器将输入数据映射到潜在空间，通过解码器将潜在空间的数据映射回原始数据空间。
  - VAE通过最大化重构损失和潜在空间的正则化项（KL散度）来学习数据的潜在表示。

- **高斯混合模型（GMM）**：
  - GMM是一种概率模型，用于将数据分布建模为多个高斯分布的混合。
  - 在本代码中，GMM用于评估VAE生成的潜在表示。

### 用途

- 该代码可用于异常声音检测任务，通过学习正常声音的潜在表示，并使用GMM评估异常声音的潜在表示，从而识别异常声音。

### 注意事项

- 确保安装了所有必要的库，包括PyTorch、Scikit-learn等。
- 确保数据路径正确，并且数据格式符合预期。
- 根据实际情况调整模型参数，如学习率、批量大小等。
- 确保GPU可用（如果使用GPU进行计算）。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from sklearn import mixture
from BaseASD.VAE.data import *
from BaseASD.VAE.model import Autoencoder
from BaseASD.VAE.common import yaml_load

if torch.cuda.is_available:
    device = "cuda:0"
else:
    device = "cpu"
print('device:' + device)

param = yaml_load()
data_path_dev = param['data_path_dev']
data_path_eval = param['data_path_eval']


def calculation_latent(ty_pe, ID, data_path_train, data_path_test):
    """
    计算潜在表示
    :param ty_pe: 数据类型
    :param ID: 数据ID
    :param data_path_train: 训练数据路径
    :param data_path_test: 测试数据路径
    :return: None
    """
    # 变量设置
    optimizer = param['optimizer']
    block = param['block']
    EPOCH_MAX = param['EPOCH_MAX']
    batch_size = param['batch_size']
    learning_rate = param['learning_rate']
    input_size = param['input_size']
    hidden1 = param['hidden1']
    hidden2 = param['hidden2']
    hidden3 = param['hidden3']
    hidden4 = param['hidden4']
    latent_length = param['latent_length']
    """
    数据加载
    """
    X_train = dnn_data_train(datadir=data_path_train, type=ty_pe, ID=ID)  # 训练数据
    # X_val = dnn_data_test(datadir=data_path, type=type, ID=ID)  # 单测试数据
    X_val, X_val_a = dnn_data_2test(datadir=data_path_test, type=ty_pe, ID=ID)  # 双测试数据

    autoencoder = Autoencoder(input_size, hidden1, hidden2, hidden3, hidden4, latent_length)
    autoencoder = autoencoder.float()
    criterion = nn.MSELoss()
    # criterion = nn.CrossEntropyLoss()
    if optimizer == 'SGD':
        optimizer = optim.SGD(autoencoder.parameters(), lr=learning_rate, momentum=0.9)  # 学习率，权重衰减
    else:
        # 实际配置的是Adam
        optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    autoencoder = autoencoder.to(device)
    criterion = criterion.to(device)

    """
    train: all true
    val: all true
    test: all anomaly
    """

    train_dataset = data.TensorDataset(torch.from_numpy(X_train))
    val_dataset = data.TensorDataset(torch.from_numpy(X_val))
    test_dataset = data.TensorDataset(torch.from_numpy(X_val_a))

    train_loader = data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    validation_loader = data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    test_loader = data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        pin_memory=True
    )

    data_loaders = {"train": train_loader, "val": validation_loader, "test": test_loader}
    # data_loaders = {"train": train_loader, "test": test_loader}
    """
    执行训练
    """
    all_running_loss = []
    all_val_loss = []
    all_test_loss = []
    all_kl_loss = []
    all_traning_latent = []
    all_val_latent = []
    all_test_latent = []
    llh1_llh1_std = []
    optimizer.zero_grad()
    gmm = mixture.GaussianMixture()
    # 初始给一个大值
    min_test_loss  = np.inf
    min_running_loss = np.inf
    for epoch in range(EPOCH_MAX):
        for phase in ['train', 'val', 'test']:
            print("Current phase is " + phase + "!")
            if phase == 'train':
                autoencoder.train()
            else:
                autoencoder.eval()
            for step, data_sample in enumerate(data_loaders[phase]):
                inputs = data_sample
                inputs = inputs[0]
                inputs = inputs.to(device).to(dtype=torch.float32)
                """
                调用一个自动编码器（Autoencoder）模型，并传入输入数据（`inputs`），然后返回模型的输出、潜在表示（latent representation）、潜在均值（latent mean）和潜在对数方差（latent log variance）。
                1. **输入数据类型转换**：`inputs.float()` 将输入数据转换为浮点数类型。这是因为在深度学习模型中，浮点数类型的数据处理通常更稳定和高效。                
                2. **调用自动编码器模型**：`autoencoder(inputs.float())` 调用自动编码器模型，并传入转换后的输入数据。自动编码器是一种神经网络结构，其目标是通过学习输入数据的压缩表示（潜在表示）来重建输入数据。
                3. **返回值**：
                   - `outputs`：自动编码器的输出，即重建的输入数据。
                   - `latent`：潜在表示，即输入数据在潜在空间中的表示。
                   - `latent_mean`：潜在均值，这是潜在空间中每个维度的均值。
                   - `latent_logvar`：潜在对数方差，这是潜在空间中每个维度的对数方差。
                
                **用途**：
                - **数据重建**：自动编码器的主要用途之一是数据重建，即通过学习输入数据的潜在表示来重建输入数据，从而实现数据去噪、压缩等任务。
                - **特征提取**：自动编码器可以用于特征提取，通过学习输入数据的潜在表示，提取出对重建输入数据最有用的特征。
                - **生成模型**：自动编码器可以用于生成模型，通过在潜在空间中采样，然后通过解码器生成新的数据。
                
                **注意事项**：
                - **数据预处理**：在调用自动编码器之前，需要对输入数据进行适当的预处理，比如归一化、标准化等。
                - **模型训练**：自动编码器需要通过训练来学习输入数据的潜在表示，训练过程中需要定义损失函数，并使用优化算法来最小化损失函数。
                - **潜在空间理解**：自动编码器的潜在空间是一个连续的、低维的空间，理解这个空间对于解释模型的行为和提取有用的特征非常重要。
                """
                outputs, latent, latent_mean, latent_logvar = autoencoder(inputs.float())
                latent_com = latent
                latent_out = latent_com.detach()
                kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
                all_kl_loss.append(kl_loss.item())
                """
                这里是损失函数，用的MSE + KL散度
                """
                loss = criterion(inputs, outputs) + kl_loss

                if phase == 'train':
                    all_running_loss.append(loss.item())  # 统计所有训练loss
                    all_traning_latent.append(latent_out.cpu().numpy())  # 统计所有训练latent

                    loss.backward()
                    optimizer.step()

                elif phase == 'val':
                    all_val_loss.append(loss.item())  # 统计所有测试正常loss
                    all_val_latent.append(latent_out.cpu().numpy())  # 统计所有测试正常latent

                else:
                    all_test_loss.append(loss.item())  # 统计所有测试异常loss
                    all_test_latent.append(latent_out.cpu().numpy())  # 统计所有测试异常latent
                optimizer.zero_grad()

        running_loss = np.mean(all_running_loss)
        # val_loss = np.mean(all_val_loss)
        test_loss = np.mean(all_test_loss)
        kl_loss = np.mean(all_kl_loss)

        """
        保存数据和模型
        """
        if os.path.exists('clustering/' + ty_pe + '/epoch/' + str(epoch)):
            pass
        else:
            # os.mkdir('clustering/' + ty_pe + '/epoch/' + str(epoch))  # 逐级创建
            os.makedirs('clustering/' + ty_pe + '/epoch/' + str(epoch))  # 一次创建到底

        # np.save('clustering/' + ty_pe + '/epoch/' + str(epoch) + '/all_traning_latent' + ID, all_traning_latent)
        # all_traning_latent = np.load('clustering/' + ty_pe + '/epoch/' + str(epoch) +
        #                              '/all_traning_latent' + ID + '.npy')
        # print('all_traning_latent.shape:', all_traning_latent.shape)

        # np.save('clustering/' + ty_pe + '/epoch/' + str(epoch) + '/all_val_latent' + ID, all_val_latent)
        # all_val_latent = np.load('clustering/' + ty_pe + '/epoch/' + str(epoch) + '/all_val_latent' + ID + '.npy')
        # print('all_val_latent.shape:', all_val_latent.shape)

        # np.save('clustering/' + ty_pe + '/epoch/' + str(epoch) + '/all_test_latent' + ID, all_test_latent)
        # all_test_latent = np.load('clustering/' + ty_pe + '/epoch/' + str(epoch) +
        #                           '/all_test_latent' + ID + '.npy')
        # print('all_test_latent.shape:', all_test_latent.shape)

        all_traning_latent = all_traning_latent.reshape(-1, 30)
        gmm.fit(all_traning_latent)
        llh1 = gmm.score_samples(all_traning_latent)
        llh1 = llh1.reshape(batch_size, -1)
        llh1_llh1_std.append(np.mean(np.std(llh1, axis=0)))

        f = open('clustering/' + ty_pe + '/llh1_llh1_std' + ID + '.txt', 'w')
        for ip in llh1_llh1_std:
            f.write(str(ip))
            f.write('\n')
        f.close()

        # 除最后一轮外每个epoch清零
        if epoch == EPOCH_MAX - 1:
            pass
        else:
            all_running_loss = []
            all_val_loss = []
            all_test_loss = []
            all_kl_loss = []
            all_traning_latent = []
            all_val_latent = []
            all_test_latent = []

        print('\n[Epoch: %3d] Train loss: %.5g Test loss: %.5g kl_loss: %.5g'
              % (epoch, float(running_loss), float(test_loss), float(kl_loss)))

        # 如果loss有减小，保存模型
        if epoch % 5 ==0 :
            if running_loss < min_running_loss and test_loss < min_test_loss:
                min_running_loss = running_loss
                min_test_loss = test_loss
                torch.save(autoencoder.state_dict(), 'clustering/' + ty_pe + '/epoch/' + str(epoch) + '/autoencoder_' + ID + '.pth')
                print('Model saved!')



if __name__ == '__main__':
    for ID in ['id_01']:
        calculation_latent('swp', ID, data_path_dev, data_path_dev)
    print('Good Luck')
