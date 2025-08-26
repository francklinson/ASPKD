import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from data import *
from sklearn import mixture
from BaseASD.VAE.model import Autoencoder

# 变量设置
EPOCH_MAX = 100
block = 'LSTM'  # GRU , LSTM 可选
optimizer = 'Adam'  # SGD , Adam 可选
dropout = 0
latent_length = 30
batch_size = 309  # 309  340
input_size = 640
hidden1 = 128
hidden2 = 128
hidden3 = 64
hidden4 = 64
learning_rate = 0.00001

if torch.cuda.is_available:
    device = "cuda:0"
else:
    device = "cpu"
# device = "cpu"
print('device:' + device)
print('EPOCH_MAX:' + str(EPOCH_MAX))

data_path_dev = r'C:\data\音频素材\异音检测\dev_data'
data_path_eval = r'C:\data\音频素材\异音检测\eval_data'


def calculation_latent(ty_pe, ID, data_path_train, data_path_test):
    print(ty_pe, ID)
    global optimizer, block, EPOCH_MAX, batch_size, learning_rate, \
        input_size, hidden1, hidden2, hidden3, hidden4, latent_length, device

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
        optimizer = optim.Adam(autoencoder.parameters(), lr=learning_rate, betas=(0.9, 0.99))
    autoencoder = autoencoder.to(device)
    criterion = criterion.to(device)

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

    for epoch in range(EPOCH_MAX):
        for phase in ['train', 'val', 'test']:
            # for phase in ['train', 'test']:
            if phase == 'train':
                autoencoder.train()
            else:
                autoencoder.eval()

            for step, data_sample in enumerate(data_loaders[phase]):
                inputs = data_sample
                inputs = inputs[0]
                inputs = inputs.to(device)
                inputs = inputs.to(torch.float32)  # 加这一行，统一一下输入和输出的数据格式
                outputs, latent, latent_mean, latent_logvar = autoencoder(inputs.float())
                latent_com = latent
                latent_out = latent_com.detach()
                kl_loss = -0.5 * torch.mean(1 + latent_logvar - latent_mean.pow(2) - latent_logvar.exp())
                all_kl_loss.append(kl_loss.item())
                """这里是损失函数，用的MSE"""
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

        if os.path.exists('clustering/mini/' + ty_pe + '/epoch/' + str(epoch)):
            pass
        else:
            # os.mkdir('clustering/mini/' + ty_pe + '/epoch/' + str(epoch))  # 逐级创建
            os.makedirs('clustering/mini/' + ty_pe + '/epoch/' + str(epoch))  # 一次创建到底

        np.save('clustering/mini/' + ty_pe + '/epoch/' + str(epoch) + '/all_traning_latent' + ID, all_traning_latent)
        all_traning_latent = np.load('clustering/mini/' + ty_pe + '/epoch/' + str(epoch) +
                                     '/all_traning_latent' + ID + '.npy')
        print('all_traning_latent.shape:', all_traning_latent.shape)

        np.save('clustering/mini/' + ty_pe + '/epoch/' + str(epoch) + '/all_val_latent' + ID, all_val_latent)
        all_val_latent = np.load('clustering/mini/' + ty_pe + '/epoch/' + str(epoch) + '/all_val_latent' + ID + '.npy')
        print('all_val_latent.shape:', all_val_latent.shape)

        np.save('clustering/mini/' + ty_pe + '/epoch/' + str(epoch) + '/all_test_latent' + ID, all_test_latent)
        all_test_latent = np.load('clustering/mini/' + ty_pe + '/epoch/' + str(epoch) +
                                  '/all_test_latent' + ID + '.npy')
        print('all_test_latent.shape:', all_test_latent.shape)

        all_traning_latent = all_traning_latent.reshape(-1, 30)
        gmm.fit(all_traning_latent)
        llh1 = gmm.score_samples(all_traning_latent)
        llh1 = llh1.reshape(batch_size, -1)
        llh1_llh1_std.append(np.mean(np.std(llh1, axis=0)))

        f = open('clustering/mini/' + ty_pe + '/llh1_llh1_std' + ID + '.txt', 'w')
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


def main():
    # for ty_pe in ['fan', 'slider', 'pump', 'valve', 'ToyCar', 'ToyConveyor']:
    for ID in ['id_01', 'id_02', 'id_03', 'id_04']:
        calculation_latent('spk', ID, data_path_dev, data_path_dev)
    print('Good Luck')


if __name__ == '__main__':
    main()
