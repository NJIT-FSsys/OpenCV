from ultralytics import YOLO

if __name__ == '__main__':
    # Create a new YOLO model from scratch
    model = YOLO('yolov8n.yaml')

    # Load a pretrained YOLO model (recommended for training)
    model = YOLO('yolov8n.pt')

    # Train the model using the 'coco128.yaml' dataset for 3 epochs
    results = model.train(data='coco128.yaml', epochs=70)

    # Evaluate the model's performance on the validation set
    results = model.val()

    # Export the model to ONNX format
    success = model.export(format='onnx')