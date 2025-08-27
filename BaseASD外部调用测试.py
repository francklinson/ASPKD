from BaseASD.DenseAE.DenseAE_interface import DenseAEInterface
from BaseASD.ConvolutionalAE.CAE_interface import CAEInterface
from BaseASD.VAE.VAE_interface import VAEInterface
from BaseASD.AEGAN.AeGan_interface import AEGANInterface
from BaseASD.DifferNet.DifferNet_interface import DifferNetInterface

if __name__ == '__main__':
    d = DenseAEInterface()
    print(d.judge_is_normal(r"C:\data\音频汇总\BC1076636\SNAPSHOT_FILE\step2_扫频-8k.wav"))
    # print(d.judge_is_normal(r"C:\data\音频汇总\BC1076636\SNAPSHOT_FILE\step2_扫频-16k.wav"))
    # print(d.judge_is_normal(r"C:\data\音频汇总\BC1076636\SNAPSHOT_FILE\step2_扫频-22.05k.wav"))
    # print(d.judge_is_normal(r"C:\data\音频汇总\BC1076636\SNAPSHOT_FILE\step2_扫频-32k.wav"))
    # print(d.judge_is_normal(r"C:\data\音频汇总\BC1076636\SNAPSHOT_FILE\step2_扫频-44.1k.wav"))
    # print(d.judge_is_normal(r"C:\data\音频汇总\BC1076636\SNAPSHOT_FILE\step2_扫频-48k.wav"))
    #
    # print(d.judge_is_normal(r"C:\data\音频汇总\TL-SPK201MP-PoEDC 1.0(修复市场反馈)_软件\1\BC1076636\SNAPSHOT_FILE\step2_扫频-8k.wav"))
    # print(d.judge_is_normal(r"C:\data\音频汇总\TL-SPK201MP-PoEDC 1.0(修复市场反馈)_软件\1\BC1076636\SNAPSHOT_FILE\step2_扫频-16k.wav"))
    # print(d.judge_is_normal(r"C:\data\音频汇总\TL-SPK201MP-PoEDC 1.0(修复市场反馈)_软件\1\BC1076636\SNAPSHOT_FILE\step2_扫频-22.05k.wav"))
    # print(d.judge_is_normal(r"C:\data\音频汇总\TL-SPK201MP-PoEDC 1.0(修复市场反馈)_软件\1\BC1076636\SNAPSHOT_FILE\step2_扫频-32k.wav"))
    # print(d.judge_is_normal(r"C:\data\音频汇总\TL-SPK201MP-PoEDC 1.0(修复市场反馈)_软件\1\BC1076636\SNAPSHOT_FILE\step2_扫频-44.1k.wav"))
    # print(d.judge_is_normal(r"C:\data\音频汇总\TL-SPK201MP-PoEDC 1.0(修复市场反馈)_软件\1\BC1076636\SNAPSHOT_FILE\step2_扫频-48k.wav"))
    # c = CAEInterface()
    # print(c.judge_is_normal(r"C:\data\音频汇总\BC1076636\SNAPSHOT_FILE\step2_扫频-8k.wav"))
    # v = VAEInterface()
    # print(v.judge_is_normal(r"C:\data\音频汇总\BC1076636\SNAPSHOT_FILE\step2_扫频-8k.wav"))
    # a = AEGANInterface()
    # print(a.judge_is_normal(r"C:\data\音频汇总\BC1076636\SNAPSHOT_FILE\step2_扫频-8k.wav"))
    # d = DifferNetInterface()
    # print(d.judge_is_normal(r"C:\Users\12165\PycharmProjects\音频异常检测\BaseASD\DifferNet\eval"))