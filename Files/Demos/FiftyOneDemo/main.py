import fiftyone.zoo as foz
import fiftyone as fo
import os;

import torch.cuda

os.environ["YOLO_VERBOSE"] = "False"
import fiftyone.utils.ultralytics as fou
from ultralytics import YOLO
from samplingstrategy import MinScore
from scoringfunction import AverageConfidence

source_path = "C:/Users/Zsombor/fiftyone/bdd100k"
NUM_OF_SAMPLES = 1200
validation_set = foz.load_zoo_dataset(
    "bdd100k",
    source_dir = source_path,
    split = "validation",
    max_samples=NUM_OF_SAMPLES
)
full_train_set = foz.load_zoo_dataset(
    "bdd100k",
    source_dir = source_path,
    split = "train",
    max_samples=NUM_OF_SAMPLES
)
test_set = foz.load_zoo_dataset(
    "bdd100k",
    source_dir = source_path,
    split = "test",
    max_samples=NUM_OF_SAMPLES
)

default_classes = ["car","traffic sign","traffic light","person","truck","bus","rider","motor","bike","train"]
session = fo.launch_app(full_train_set, auto=False)
print(session)
EXPORT_DIR = "C:/Users/Zsombor/Documents/Paripa/PARIPA_Active_Learning/Files/Demos/FiftyOneDemo/Yolo/"
dataset_type=fo.types.YOLOv5Dataset

# fo.delete_dataset("bdd100k-training_set")
# fo.delete_dataset("bdd100k-pool")
'''
INITIAL_TRAINING_SIZE = 300

train_set = fo.Dataset("bdd100k-training_set",persistent=True)
pool = fo.Dataset("bdd100k-pool",persistent=True)

for idx, sample in enumerate(full_train_set):
    if idx < INITIAL_TRAINING_SIZE:
        train_set.add_sample(sample=sample)
    else:
        pool.add_sample(sample=sample)

train_set.default_classes = default_classes
pool.default_classes = default_classes
'''


def export_sets(_dir=EXPORT_DIR):
    validation_set.export(
        export_dir=_dir,
        dataset_type=dataset_type,
        label_field="detections",
        split="val",  # Ultralytics uses 'val'
        classes=default_classes,
    )

    train_set.export(
        export_dir=_dir,
        dataset_type=dataset_type,
        classes=default_classes,
        label_field="detections",
        split="train",
    )


def predict(_dataset, _model, _prediction_tag: str = "predictions", _iou=0.5):
    for sample in _dataset.iter_samples(progress=True):
        result = _model.predict(sample.filepath, iou=_iou)[0]
        sample[_prediction_tag] = fou.to_detections(result)
        sample.save()


train_set = fo.load_dataset("bdd100k-training_set")
train_set.default_classes = default_classes
pool = fo.load_dataset("bdd100k-pool")
pool.default_classes = default_classes


print(torch.cuda.is_available())

def active_learning_loop():
    model = YOLO("yolov8s.pt")
    for i in range(1):
        print("predicting:")
        predict(pool, model)
        print("querying:")
        pool, data_to_query = MinScore().get_n_samples(_pool=pool, _scoring_function=AverageConfidence(), _n_samples=100)
        train_set.add_samples(data_to_query)
        print("exporting sets:")
        export_sets(EXPORT_DIR + str(i))
        print("training model:")
        YAML_FILE = EXPORT_DIR + str(i) + "/dataset.yaml"
        model.train(data=YAML_FILE, epochs=3)

        print("" + str(i) + " active learning iteration ended")
    model.export(format="onnx")


if torch.cuda.is_available():
    torch.cuda.set_device(0)

trained_model = YOLO("C:/Users/Zsombor/Documents/Paripa/PARIPA_Active_Learning/runs/detect/train4/weights/best.onnx")


#predict(pool, trained_model)

trained_model.predict("C:/Users/Zsombor/Documents/Paripa/yolov7/imgs/0.jpg")


results = pool.evaluate_detections(
    "predictions",
    gt_field="detections",
    eval_key="eval_predictions"
)

results.print_report(classes = default_classes)

input()