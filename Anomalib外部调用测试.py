from Anomalib.data import MVTecAD, Folder
from Anomalib.models import Patchcore, Csflow, Cfa, Cflow, VlmAd, WinClip, Dfkde, Dfm, Draem, Dsr, EfficientAd, \
    Fastflow, Fre, Ganomaly, Padim, ReverseDistillation, Stfpm, Supersimplenet, Uflow, Dinomaly
from Anomalib.engine import Engine
from Anomalib.data import PredictDataset
from Anomalib.visualization import visualize_image_item
from pathlib import Path

"""
Image Models:
    - CFA (:class:`Anomalib.models.image.Cfa`)
    - Cflow (:class:`Anomalib.models.image.Cflow`)
    - CSFlow (:class:`Anomalib.models.image.Csflow`)
    - DFKDE (:class:`Anomalib.models.image.Dfkde`)
    - DFM (:class:`Anomalib.models.image.Dfm`)
    - DRAEM (:class:`Anomalib.models.image.Draem`)
    - DSR (:class:`Anomalib.models.image.Dsr`)
    - EfficientAd (:class:`Anomalib.models.image.EfficientAd`)
    - FastFlow (:class:`Anomalib.models.image.Fastflow`)
    - FRE (:class:`Anomalib.models.image.Fre`)
    - GANomaly (:class:`Anomalib.models.image.Ganomaly`)
    - PaDiM (:class:`Anomalib.models.image.Padim`)
    - PatchCore (:class:`Anomalib.models.image.Patchcore`)
    - Reverse Distillation (:class:`Anomalib.models.image.ReverseDistillation`)
    - STFPM (:class:`Anomalib.models.image.Stfpm`)
    - SuperSimpleNet (:class:`Anomalib.models.image.Supersimplenet`)
    - UFlow (:class:`Anomalib.models.image.Uflow`)
    - VLM-AD (:class:`Anomalib.models.image.VlmAd`)
    - WinCLIP (:class:`Anomalib.models.image.WinClip`)

"""

# Initialize components
datamodule = MVTecAD(root="data/spk", category="IP", )

# datamodule = Folder(
#     name="spk",
#     normal_dir="data/spk/TRY/train/good",
#     normal_test_dir="data/spk/TRY/test/good",
#     abnormal_dir="data/spk/TRY/test/bad",
#     train_batch_size=32,
#     eval_batch_size=32,
# )
datamodule.setup()

print(datamodule.task)
model = Fastflow()
engine = Engine(max_epochs=200)


def train():
    # Train the model
    engine.train(datamodule=datamodule, model=model)


def model_test():
    # Load model and make predictions
    predictions = engine.predict(
        datamodule=datamodule,
        model=model,
        ckpt_path="results/Supersimplenet/MVTecAD/N32/v0/weights/lightning/model.ckpt",
    )
    if predictions is not None:
        for prediction in predictions:
            image_path = prediction.image_path
            anomaly_map = prediction.anomaly_map  # Pixel-level anomaly heatmap
            pred_label = prediction.pred_label  # Image-level label (0: normal, 1: anomalous)
            pred_score = prediction.pred_score  # Image-level anomaly score
            gt_label = prediction.gt_label
            print(pred_label)
            print(pred_score)
            print(gt_label)
            # 计算pred_label和gt_label之间的正确率
            cnt = 0
            for i in range(len(pred_label)):
                if pred_label[i] == gt_label[i]:
                    cnt += 1
            print("正确率：", cnt / len(pred_label))

    return predictions


if __name__ == '__main__':
    train()
    # model_test()
