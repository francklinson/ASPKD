"""
这段Python代码的主要功能是计算并绘制不同类型的机器学习模型在多个epoch下的AUC（Area Under Curve）和pAUC（Partial AUC）值，
并保存这些图表。具体来说，它使用高斯混合模型（Gaussian Mixture Model, GMM）来对训练、验证和测试数据集进行拟合和评分，
然后计算这些评分的均值和标准差，并使用这些值来计算AUC和pAUC。最后，它将这些计算结果绘制成图表并保存。

以下是代码的详细解释：

1. **导入必要的库**：
   - `os`：用于处理文件和目录。
   - `numpy`：用于数值计算。
   - `matplotlib.pyplot`：用于绘制图表。
   - `sklearn.metrics.roc_auc_score`：用于计算AUC和pAUC。
   - `sklearn.mixture`：用于高斯混合模型。

2. **定义常量**：
   - `EPOCH_MAX`：最大epoch数，即训练的轮数。
   - `batch`：批次大小。

3. **定义函数`calculation_auc`**：
   - 该函数接受三个参数：`ty_pe`（数据类型），`ID`（数据集ID），`batch`（批次大小）。
   - `p_auc`、`auc`、`llh1_llh1_std`：分别用于存储每个epoch的pAUC、AUC和标准差。
   - 在每个epoch中，加载训练、验证和测试数据的潜在表示（latent representation），并使用高斯混合模型对训练数据进行拟合。
   - 计算训练、验证和测试数据的评分，并计算它们的均值和标准差。
   - 使用这些评分计算pAUC和AUC。
   - 绘制pAUC、AUC和标准差的图表，并保存。

4. **主程序**：
   - 遍历不同的数据类型（`ty_pe`），并调用`calculation_auc`函数计算和绘制图表。

**注意事项**：
- 代码中使用了硬编码的文件路径和文件名，这可能会导致在不同环境下运行时出现问题。建议使用相对路径或配置文件来管理这些路径。
- 代码中使用了`os.makedirs`来创建目录，这会创建所有必要的中间目录，如果目录已经存在，则不会报错。
- 代码中使用了`plt.savefig`来保存图表，但注释掉了保存的路径，如果需要保存图表，请取消注释并确保路径存在。
- 代码中的`roc_auc_score`函数使用了`max_fpr=0.1`参数，这表示只考虑假阳性率小于0.1的部分来计算AUC。

"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn import mixture

EPOCH_MAX = 100
batch = 309


def calculation_auc(ty_pe, ID, batch):
    """
    使用高斯混合模型（Gaussian Mixture Model, GMM）对训练、验证和测试数据集进行聚类，并计算一些评估指标。
    1. **循环遍历多个epoch**：代码使用一个for循环遍历多个epoch，每个epoch都会进行一次高斯混合模型的训练和评估。
    2. **加载数据**：在每个epoch中，代码从文件中加载训练、验证和测试数据的潜在表示（latent representations）。这些数据存储在`all_traning_latent`、`all_val_latent`和`all_test_latent`变量中。
    3. **重塑数据**：将加载的数据重塑为形状为(-1, 30)的二维数组，其中-1表示自动计算行数，30表示列数。
    4. **训练高斯混合模型**：使用训练数据拟合高斯混合模型。
    5. **计算对数似然**：对训练、验证和测试数据计算高斯混合模型的对数似然值。
    6. **重塑对数似然值**：将计算得到的对数似然值重塑为形状为(batch, -1)的二维数组。
    7. **计算对数似然差值**：计算训练数据的对数似然平均值与验证和测试数据的对数似然平均值之间的差值。
    8. **生成标签和预测值**：根据验证和测试数据的形状生成标签（y_true）和预测值（y_pred）。
    9. **计算AUC和P-AUC**：使用ROC曲线下面积（AUC）和部分ROC曲线下面积（P-AUC）评估模型性能。
    10. **计算对数似然标准差**：计算训练数据的对数似然标准差。

    注意事项：
    - `roc_auc_score`函数来自`sklearn.metrics`模块，需要先导入。
    - `cdist`函数用于计算两个数组之间的距离，需要先导入`scipy.spatial.distance`模块。
    - `mixture.GaussianMixture`是`sklearn.mixture`模块中的一个类，用于实现高斯混合模型。
    """
    p_auc = []
    auc = []
    llh1_llh1_std = []
    for epoch in range(EPOCH_MAX):
        gmm = mixture.GaussianMixture()

        all_traning_latent = np.load( 'clustering/' + ty_pe + '/epoch/' + str(epoch) + '/all_traning_latent' + ID + '.npy')
        all_val_latent = np.load('clustering/' + ty_pe + '/epoch/' + str(epoch) + '/all_val_latent' + ID + '.npy')
        all_test_latent = np.load('clustering/' + ty_pe + '/epoch/' + str(epoch) + '/all_test_latent' + ID + '.npy')

        all_traning_latent = all_traning_latent.reshape(-1, 30)
        all_val_latent = all_val_latent.reshape(-1, 30)
        all_test_latent = all_test_latent.reshape(-1, 30)

        # train1_train2_latent.append(np.mean(cdist(all_traning_latent, all_traning_latent, metric='euclidean')))
        gmm.fit(all_traning_latent)
        llh1 = gmm.score_samples(all_traning_latent)
        llh2 = gmm.score_samples(all_val_latent)
        llh3 = gmm.score_samples(all_test_latent)
        llh1 = llh1.reshape(batch, -1)
        llh2 = llh2.reshape(batch, -1)
        llh3 = llh3.reshape(batch, -1)

        llh1_llh2 = np.mean(llh1) - np.mean(llh2, axis=0)
        llh1_llh3 = np.mean(llh1) - np.mean(llh3, axis=0)
        # all_val_latent对应的是正常的数据，all_test_latent对应的是异常的数据
        y_true = [0] * (all_val_latent.shape[0] // batch) + [1] * (all_test_latent.shape[0] // batch)
        y_pred = np.concatenate((llh1_llh2, llh1_llh3), axis=0)
        y_pred = np.array(y_pred)
        p_auc.append(roc_auc_score(y_true, y_pred, max_fpr=0.1))
        auc.append(roc_auc_score(y_true, y_pred))
        llh1_llh1_std.append(np.mean(np.std(llh1, axis=0)))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    font0 = {'family': 'Times New Roman', 'weight': 'normal', 'style': 'italic', 'size': 18}
    font1 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 12}
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15}
    font3 = {'family': 'Times New Roman', 'weight': 'normal', 'style': 'italic', 'size': 15}
    ax2 = ax.twinx()
    lns1 = ax.plot(p_auc, label='p_auc', color='b')
    lns2 = ax.plot(auc, label='auc', color='y')
    lns3 = ax2.plot(llh1_llh1_std, color='g', linestyle='--', label=r'std(likelihood)')
    ax.set_xlabel('Epoch', font2)
    ax.set_ylabel('AUC/pAUC', font2)
    ax2.set_ylabel('std(likelihood)', font3)
    plt.title(ty_pe + '_' + ID, font2)
    lns = lns1 + lns2 + lns3
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=0, prop=font1)
    ax.tick_params(labelsize=14)
    ax2.tick_params(labelsize=14)
    plt.tight_layout()
    if os.path.exists('big_sample/' + ty_pe):
        pass
    else:
        os.makedirs('big_sample/' + ty_pe)
    plt.savefig('big_sample/' + ty_pe + "/_" + ID + ".png")
    plt.close()


if __name__ == '__main__':
    for ID in ['id_01']:
        calculation_auc('swp', ID, batch)
    print('Good Luck')
