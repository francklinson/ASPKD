"""
Script for test
"""
import os
import sys
import numpy
import BaseASD.ConvolutionalAE.common as com
from BaseASD.ConvolutionalAE import keras_model


class CAEInterface:
    def __init__(self):
        self.param = com.yaml_load()
        self.model = None
        self.load_model()

    def load_model(self):
        """
        load model
        """
        machine_type = "spk"
        print("============== MODEL LOAD ==============")
        # set model path
        model_file = "BaseASD/ConvolutionalAE/model/model_spk.hdf5"
        # load model file
        if not os.path.exists(model_file):
            com.logger.error("{} model not found ".format(machine_type))
            # sys.exit(-1)
        self.model = keras_model.load_model(model_file)

    @staticmethod
    def check_file_path(input_file_path):
        """
        检查输入的音频文件格式是否正确
        """
        if not os.path.exists(input_file_path):
            com.logger.error("file not found!!: {}".format(input_file_path))
            sys.exit(-1)
        # 确定是文件
        if not os.path.isfile(input_file_path):
            com.logger.error("not file!!: {}".format(input_file_path))
            sys.exit(-1)
        # 检查文件格式，只支持wav
        if not input_file_path.lower().endswith(".wav"):
            com.logger.error("file not wav!!: {}".format(input_file_path))
            sys.exit(-1)

    def predict(self, file_path, ):
        try:
            # get audio features
            vector_array = com.file_to_vector_array(file_path,
                                                    n_mels=self.param["feature"]["n_mels"],
                                                    frames=self.param["feature"]["frames"],
                                                    n_fft=self.param["feature"]["n_fft"],
                                                    hop_length=self.param["feature"]["hop_length"],
                                                    power=self.param["feature"]["power"])
            length, _ = vector_array.shape
            dim = self.param["autoencoder"]["shape0"]
            step = self.param["step"]
            idex = numpy.arange(length - dim + step, step=step)
            batch = None
            for idx in range(len(idex)):
                start = min(idex[idx], length - dim)
                vector = vector_array[start:start + dim, :]
                vector = vector.reshape((1, vector.shape[0], vector.shape[1]))
                if idx == 0:
                    batch = vector
                else:
                    batch = numpy.concatenate((batch, vector))
            # add channels dimension
            data = batch.reshape((batch.shape[0], batch.shape[1], batch.shape[2], 1))

            # calculate predictions
            errors = numpy.mean(numpy.square(data - self.model.predict(data)), axis=-1)
            anomaly_score = numpy.mean(errors)
            return anomaly_score
        except:
            com.logger.error("file broken!!: {}".format(file_path))
            return 0

    def judge_is_normal(self, file_path, ):
        """
        判断是否为正常音频
        """
        self.check_file_path(file_path)
        anomaly_score = self.predict(file_path)
        if anomaly_score < self.param["anomaly_threshold"]:
            return True, anomaly_score
        else:
            return False, anomaly_score


if __name__ == "__main__":
    file_path = r"E:\音频素材\异音检测\dev_data\spk\test\anomaly_id_02_00000001.wav"
    cae = CAEInterface()
    print(cae.judge_is_normal(file_path))
