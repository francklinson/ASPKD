"""定义ASD基础类"""
import os
import sys


class AnomalySoundDetectionBase(object):
    def init(self):
        pass

    def load_model(self):
        raise NotImplementedError("Need to implement load_model function !!!")

    @staticmethod
    def check_file_path(input_file_path):
        """
        检查输入的音频文件格式是否正确
        """
        if not os.path.exists(input_file_path):
            print("file not found!!: {}".format(input_file_path))
            sys.exit(-1)
        # 确定是文件
        if not os.path.isfile(input_file_path):
            print("not file!!: {}".format(input_file_path))
            sys.exit(-1)
        # 检查文件格式，只支持wav
        if not input_file_path.lower().endswith(".wav"):
            print("file not wav!!: {}".format(input_file_path))
            sys.exit(-1)

    def predict(self, input_file_path):
        raise NotImplementedError("Need to implement predict function !!!")

    def judge_is_normal(self, file_path):
        raise NotImplementedError("Need to implement judge_is_normal function !!!")
