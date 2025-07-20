import pickle
from ultralytics import YOLO


def yolo_subprocess_batch(model_file_name, batch_images):
    model_instance = YOLO(model_file_name)

    model_results = model_instance(batch_images, device='cpu')

    # Save results to a temporary file
    with open('/tmp/yolo_results.pkl', 'wb') as f:
        pickle.dump(model_results, f)


def yolo_subprocess_val(model_file_name, config):
    model_instance = YOLO(model_file_name)
    metrics = model_instance.val(data=config, device='cpu')

    # Save results to a temporary file
    with open('/tmp/yolo_results.pkl', 'wb') as f:
        pickle.dump(metrics, f)