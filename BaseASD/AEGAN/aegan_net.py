"""
定义网络结构
用于生成对抗网络（GAN）的编码器（Encoder）、解码器（Decoder）和判别器（Discriminator）
"""
import torch.nn as nn


def ActLayer(act):
    assert act in ['relu', 'leakyrelu', 'tanh'], 'Unknown activate function!'
    if act == 'relu':
        return nn.ReLU(True)
    elif act == 'leakyrelu':
        return nn.LeakyReLU(0.2, True)
    elif act == 'tanh':
        return nn.Tanh()
    return None


def NormLayer(normalize, chan, reso):
    assert normalize in ['bn', 'ln', 'in'], 'Unknown normalize function!'
    if normalize == 'bn':
        return nn.BatchNorm2d(chan)
    elif normalize == 'ln':
        return nn.LayerNorm((chan, reso, reso))
    elif normalize == 'in':
        return nn.InstanceNorm2d(chan)
    return None


class DCEncoder(nn.Module):
    """
    DCGAN DCEncoder NETWORK
    DCEncoder类定义了一个编码器网络，用于将输入图像编码为潜在向量。
    网络结构包括卷积层、归一化层和激活函数，逐步减小图像的空间尺寸并增加特征图的深度。
    最后，可以选择添加一个卷积层将特征图转换为潜在向量。
    """

    def __init__(self, isize, nz, ndf, act, normalize, add_final_conv=True):
        super(DCEncoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        main = []
        main.append(nn.Conv2d(1, ndf, 4, 2, 1, bias=False))
        main.append(NormLayer(normalize, ndf, isize // 2))
        main.append(ActLayer(act))
        csize, cndf = isize // 2, ndf

        while csize > 4:
            in_chan = cndf
            out_chan = cndf * 2
            main.append(nn.Conv2d(in_chan, out_chan, 4, 2, 1, bias=False))
            cndf = cndf * 2
            csize = csize // 2
            main.append(NormLayer(normalize, out_chan, csize))
            main.append(ActLayer(act))

        # state size. K x 4 x 4
        if add_final_conv:
            main.append(nn.Conv2d(cndf, nz, 4, 1, 0, bias=False))

        self.main = nn.Sequential(*main)

    def forward(self, x):
        z = self.main(x)
        return z


class DCDecoder(nn.Module):
    """
    DCGAN DCDecoder NETWORK
    DCDecoder类定义了一个解码器网络，用于将潜在向量解码为图像。
    网络结构包括反卷积层、归一化层和激活函数，逐步增加图像的空间尺寸并减少特征图的深度。
    最后，使用Tanh激活函数将特征图转换为图像。
    """

    def __init__(self, isize, nz, ngf, act, normalize):
        super(DCDecoder, self).__init__()
        assert isize % 16 == 0, "isize has to be a multiple of 16"

        cngf, tisize = ngf // 2, 4
        while tisize != isize:
            cngf = cngf * 2
            tisize = tisize * 2

        main = []
        main.append(nn.ConvTranspose2d(nz, cngf, 4, 1, 0, bias=False))
        csize = 4
        main.append(NormLayer(normalize, cngf, csize))
        main.append(ActLayer(act))

        while csize < isize // 2:
            main.append(nn.ConvTranspose2d(cngf, cngf // 2, 4, 2, 1, bias=False))
            cngf = cngf // 2
            csize = csize * 2
            main.append(NormLayer(normalize, cngf, csize))
            main.append(ActLayer(act))

        main.append(nn.ConvTranspose2d(cngf, 1, 4, 2, 1, bias=False))
        main.append(ActLayer('tanh'))
        self.main = nn.Sequential(*main)

    def forward(self, z):
        x = self.main(z)
        return x


class AEDC(nn.Module):
    """
    AEDC类定义了一个自编码器，由编码器和解码器组成。
    forward方法中，输入数据首先通过编码器转换为潜在向量，然后通过解码器重建为图像
    """
    def __init__(self, param):
        super(AEDC, self).__init__()
        self.Encoder = DCEncoder(isize=param['net']['isize'],
                                 nz=param['net']['nz'],
                                 ndf=param['net']['ndf'],
                                 act=param['net']['act'][0],
                                 normalize=param['net']['normalize']['g'])
        self.Decoder = DCDecoder(isize=param['net']['isize'],
                                 nz=param['net']['nz'],
                                 ngf=param['net']['ngf'],
                                 act=param['net']['act'][1],
                                 normalize=param['net']['normalize']['g'])

    def forward(self, data, outz=False):
        z = self.Encoder(data)
        if outz:
            return z
        else:
            recon = self.Decoder(z)
            return recon


class Discriminator(nn.Module):
    """
    Discriminator类定义了一个判别器网络，用于判断输入图像是真实图像还是生成图像。
    网络结构包括卷积层、归一化层和激活函数，逐步减小图像的空间尺寸并增加特征图的深度。
    最后，使用一个全局池化层和全连接层将特征图转换为判别结果。
    """
    def __init__(self, param):
        super(Discriminator, self).__init__()
        ndf, isize = param['net']['ndf'], param['net']['isize']
        act, normalize = param['net']['act'][0], param['net']['normalize']['d']

        self.main = nn.ModuleList()
        level = 0
        in_chan = 1
        chans, resoes = [in_chan], [isize]
        init_layer = nn.Sequential(nn.Conv2d(in_chan, ndf, 4, 2, 1, bias=False),
                                   NormLayer(normalize, ndf, isize // 2),
                                   ActLayer(act))
        level, csize, cndf = 1, isize // 2, ndf
        self.main.append(init_layer)
        chans.append(ndf)
        resoes.append(csize)

        while csize > 4:
            in_chan = cndf
            out_chan = cndf * 2
            pyramid = [nn.Conv2d(in_chan, out_chan, 4, 2, 1, bias=False)]
            level, cndf, csize = level + 1, cndf * 2, csize // 2
            pyramid.append(NormLayer(normalize, out_chan, csize))
            pyramid.append(ActLayer(act))
            self.main.append(nn.Sequential(*pyramid))
            chans.append(out_chan)
            resoes.append(csize)

        in_chan = cndf
        # 判断真假
        self.feat_extract_layer = nn.Sequential(nn.Conv2d(in_chan, in_chan, 4, 1, 0, bias=False, groups=in_chan),
                                                # GDConv
                                                nn.Flatten())  # D网络的embedding
        self.output_layer = nn.Sequential(nn.LayerNorm(in_chan),
                                          ActLayer(act),
                                          nn.Linear(in_chan, 1))

    def forward(self, x):
        for module in self.main:
            x = module(x)
        feat = self.feat_extract_layer(x)
        pred = self.output_layer(feat)
        return pred, feat

    """
    网络中的卷积和反卷积操作都使用了步长为2的卷积核，以有效地减小或增加图像的空间尺寸。
    归一化层的选择和位置对网络的训练和性能有重要影响。
    激活函数的选择也会影响网络的训练和性能，ReLU和LeakyReLU是常用的激活函数。
    判别器中的特征提取层使用了分组卷积（Grouped Convolution），这是一种特殊的卷积操作，可以减少计算量和参数数量。
    """
