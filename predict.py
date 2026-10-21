import cv2
import torch

from models.experimental import attempt_load
from utils.datasets import LoadImages, norm_imgs
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device


MODEL_PATH = "weights/esod_yolov5m.pt"
IMAGE_PATH = "VisDrone/VisDrone2019-DET-train/images/0000002_00005_d_0000014_masked.jpg"


@torch.no_grad()
def main():
    device = "cuda:0"
    model = attempt_load(MODEL_PATH, map_location=device)
    model.eval()
    names = model.module.names if hasattr(model, "module") else model.names
    stride = int(model.stride.max())

    dataset = LoadImages(IMAGE_PATH, img_size=1536, stride=stride)
    path, image, original_image, _ = next(iter(dataset))

    image = torch.from_numpy(image).to(device).float().unsqueeze(0)
    image = norm_imgs(image, model)
    (prediction, _), _ = model(image)
    detections = non_max_suppression(prediction, 0.25, 0.45)[0]

    if len(detections):
        detections[:, :4] = scale_coords(
            image.shape[2:], detections[:, :4], original_image.shape
        ).round()

    for x1, y1, x2, y2, confidence, class_id in detections.tolist():
        label = f"{names[int(class_id)]} {confidence:.2f}"
        cv2.rectangle(original_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            original_image,
            label,
            (int(x1), max(int(y1) - 5, 15)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1,
        )

    cv2.imwrite("prediction.jpg", original_image)

    print(f"Weight: {MODEL_PATH}")
    print(f"Image: {path}")
    print("Saved: prediction.jpg")
    print("Detections:")
    for x1, y1, x2, y2, confidence, class_id in detections.tolist():
        print(
            f"  {names[int(class_id)]}: confidence={confidence:.3f}, "
            f"box=({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f})"
        )
    print(f"\nNumber of predictions: {len(detections.tolist())}")


if __name__ == "__main__":
    main()
