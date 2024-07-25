import pathlib

from ultralytics import YOLO


DATASET_FOLDER = (
  pathlib.Path(__file__).resolve().parent.parent / "door-windows-1"
)


def train():
  # Create a new YOLO model from scratch
  model = YOLO("yolov8n.yaml")

  # Load a pretrained YOLO model (recommended for training)
  # model = YOLO()

  # Train the model using the 'coco8.yaml' dataset for 3 epochs
  _ = model.train(data=DATASET_FOLDER / "data.yaml", epochs=3)

  # Evaluate the model's performance on the validation set
  _ = model.val()

  # Perform object detection on an image using the model
  # results = model("https://ultralytics.com/images/bus.jpg")

  # Export the model to ONNX format
  # success = model.export(format="onnx")


if __name__ == "__main__":
  train()
