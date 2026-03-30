


from .base_processor import MusicProcessor
import abc


class MusicProcessorPredict(MusicProcessor):

    @abc.abstractmethod
    def predict_music(self, music_path, connector):
        raise NotImplementedError(u"出错了，你没有实现predict_music抽象方法")


    pass