import cv2
import numpy as np
from ultralytics import YOLO  # yolov12 via ultralytics

def generate_yolo_vegetation_mask(
    image_path, 
    model_path="yolov12_vegetation.pt", 
    save=True, 
    conf_threshold=0.5,
    mask_save_path="yolo_vegetation_mask.png"
):
    """
    Generate a vegetation mask from an RGB image using YOLOv12.
    Only class 0 (vegetation) is considered.
    Args:
        image_path (str): Path to the RGB image.
        model_path (str): Path to the YOLOv12 weights.
        save (bool): Whether to save the mask.
        conf_threshold (float): Confidence threshold for detections.
        mask_save_path (str): Where to save the mask.
    Returns:
        np.ndarray: Binary vegetation mask (255=vegetation, 0=background).
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]

    # Load YOLOv12 model
    model = YOLO(model_path)
    results = model(image)[0]  # Get first (and only) result

    # Create a blank mask
    veg_mask = np.zeros((height, width), dtype=np.uint8)

    # Loop through detections
    for box in results.boxes.data.cpu().numpy():
        x1, y1, x2, y2, conf, cls = box[:6]
        if int(cls) == 0 and conf >= conf_threshold:  # class 0 = vegetation
            x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
            cv2.rectangle(veg_mask, (x1, y1), (x2, y2), 255, -1)  # fill vegetation area

    if save:
        cv2.imwrite(mask_save_path, veg_mask)

    return veg_mask

# Example usage
if __name__ == "__main__":
    mask = generate_yolo_vegetation_mask(r"D:\AAU internship\CWF-788\IMAGE400x300 - Copy\test\401_image.jpg")
    cv2.imshow("YOLOv12 Vegetation Mask", mask)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
