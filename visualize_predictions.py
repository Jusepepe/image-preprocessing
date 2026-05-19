import os
import cv2
from ultralytics import YOLO


def main():
    # Paths
    model_path = os.path.join("models", "model_vn3.pt")
    images_dir = r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\clahe_leaves"

    # Load the model
    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # Get all images
    if not os.path.exists(images_dir):
        print(f"Error: Directory {images_dir} does not exist.")
        return

    image_files = [
        f
        for f in os.listdir(images_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    if not image_files:
        print(f"No images found in {images_dir}")
        return

    print(f"Found {len(image_files)} images.")

    print(
        "Running predictions and visualizing. Press any key to see the next image. Press 'q' to quit."
    )

    for image_file in image_files:
        img_path = os.path.join(images_dir, image_file)

        # Run YOLO prediction
        results = model.predict(
            source=img_path, save=False, show=False, iou=0.9, conf=0.15
        )

        # Get the annotated image array
        annotated_img = results[0].plot()
        print(results[0].boxes.shape)

        # Display the image
        cv2.imshow(
            "YOLOv8 Prediction",
            cv2.resize(
                annotated_img, (0, 0), fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA
            ),
        )

        # Wait for a key press (0 means wait indefinitely)
        key = cv2.waitKey(0) & 0xFF

        # If 'q' is pressed, exit the loop
        if key == ord("q"):
            break

    cv2.destroyAllWindows()
    print("Visualization finished.")


if __name__ == "__main__":
    main()
