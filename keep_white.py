import cv2
import os
import numpy as np
from clahe_test import apply_clahe


def nothing(x):
    pass


def create_trackbars(window_name):
    # LAB Trackbars
    cv2.createTrackbar("L lower", window_name, 153, 255, nothing)
    cv2.createTrackbar("A lower", window_name, 122, 255, nothing)
    cv2.createTrackbar("B lower", window_name, 120, 255, nothing)

    cv2.createTrackbar("L upper", window_name, 255, 255, nothing)
    cv2.createTrackbar("A upper", window_name, 135, 255, nothing)
    cv2.createTrackbar("B upper", window_name, 179, 255, nothing)

    # HSV Trackbars
    cv2.createTrackbar("H lower", window_name, 0, 179, nothing)
    cv2.createTrackbar("S lower", window_name, 5, 255, nothing)
    cv2.createTrackbar("V lower", window_name, 150, 255, nothing)

    cv2.createTrackbar("H upper", window_name, 134, 179, nothing)
    cv2.createTrackbar("S upper", window_name, 68, 255, nothing)
    cv2.createTrackbar("V upper", window_name, 255, 255, nothing)


def get_trackbar_values(window_name):
    """Returns lower_lab, upper_lab, lower_hsv, upper_hsv arrays."""
    l_l = cv2.getTrackbarPos("L lower", window_name)
    a_l = cv2.getTrackbarPos("A lower", window_name)
    b_l = cv2.getTrackbarPos("B lower", window_name)

    l_u = cv2.getTrackbarPos("L upper", window_name)
    a_u = cv2.getTrackbarPos("A upper", window_name)
    b_u = cv2.getTrackbarPos("B upper", window_name)

    lower_lab = np.array([l_l, a_l, b_l])
    upper_lab = np.array([l_u, a_u, b_u])

    h_l = cv2.getTrackbarPos("H lower", window_name)
    s_l = cv2.getTrackbarPos("S lower", window_name)
    v_l = cv2.getTrackbarPos("V lower", window_name)

    h_u = cv2.getTrackbarPos("H upper", window_name)
    s_u = cv2.getTrackbarPos("S upper", window_name)
    v_u = cv2.getTrackbarPos("V upper", window_name)

    lower_hsv = np.array([h_l, s_l, v_l])
    upper_hsv = np.array([h_u, s_u, v_u])

    return lower_lab, upper_lab, lower_hsv, upper_hsv


def draw_recursive_bboxes(mask, draw_img):
    """
    Finds contours and recursively splits bounding boxes if the mask
    occupies less than 75% of the bounding box area.
    """
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Calculate a dynamic thickness based on image height so bboxes are always visible after resizing
    h_img = draw_img.shape[0]
    thickness = max(2, int(h_img / 400))

    def process_bbox(x, y, w, h):
        if w <= 0 or h <= 0:
            return
        sub_mask = mask[y : y + h, x : x + w]
        area = cv2.countNonZero(sub_mask)
        if area == 0:
            return

        bbox_area = w * h
        if bbox_area < 150:
            return

        # Stop if segment fills >75% of bbox, or if bbox is too small to divide usefully
        if area / bbox_area >= 0.75 or w < 5 or h < 5:
            cv2.rectangle(draw_img, (x, y), (x + w, y + h), (0, 255, 0), thickness)
        else:
            w1, h1 = w // 2, h // 2
            w2, h2 = w - w1, h - h1
            process_bbox(x, y, w1, h1)
            process_bbox(x + w1, y, w2, h1)
            process_bbox(x, y + h1, w1, h2)
            process_bbox(x + w1, y + h1, w2, h2)

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        process_bbox(x, y, w, h)


def create_display_grid(
    original_img,
    blur_img,
    hsv_mask,
    lab_mask,
    result_mask,
    res,
    current_idx,
    total_images,
):
    """Resizes and stacks all images into a 2x3 grid for display."""
    # Convert masks to BGR for stacking
    hsv_mask_bgr = cv2.cvtColor(hsv_mask, cv2.COLOR_GRAY2BGR)
    lab_mask_bgr = cv2.cvtColor(lab_mask, cv2.COLOR_GRAY2BGR)
    result_mask_bgr = cv2.cvtColor(result_mask, cv2.COLOR_GRAY2BGR)

    # Resize for display to avoid windows that are too large
    h, w = original_img.shape[:2]
    target_h = 400
    target_w = int(w * (target_h / h))

    disp_orig = cv2.resize(blur_img, (target_w, target_h))
    disp_hsv_mask = cv2.resize(hsv_mask_bgr, (target_w, target_h))
    disp_lab_mask = cv2.resize(lab_mask_bgr, (target_w, target_h))
    disp_result_mask = cv2.resize(result_mask_bgr, (target_w, target_h))
    disp_overlay = cv2.resize(res, (target_w, target_h))

    blank = np.zeros_like(disp_orig)

    # Put text for instructions
    cv2.putText(
        disp_orig,
        f"Img {current_idx + 1}/{total_images} (CLAHE)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        disp_orig,
        "n: Next | p: Prev | q: Quit",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        disp_hsv_mask,
        "HSV Mask",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        disp_lab_mask,
        "LAB Mask",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        disp_result_mask,
        "Result Mask (AND)",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        disp_overlay,
        "Mask Overlay",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )

    # Create 2x3 grid
    row1 = np.hstack((disp_orig, disp_hsv_mask, disp_lab_mask))
    row2 = np.hstack((disp_result_mask, disp_overlay, blank))
    stacked = np.vstack((row1, row2))

    return stacked


def main():
    images_path = r"E:\fabian\pest-monitoring\vision\datasets\pest_disease_fruit_detection\whiteflies\foreground"
    images = [
        f
        for f in os.listdir(images_path)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]

    window_name = "Visualizer"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 800)

    create_trackbars(window_name)

    current_idx = 0
    while current_idx < len(images):
        image_name = images[-current_idx]
        img_path = os.path.join(images_path, image_name)
        original_img = cv2.imread(img_path)
        if original_img is None:
            current_idx += 1
            continue

        clahe_img = apply_clahe(original_img.copy())
        blur_img = cv2.medianBlur(clahe_img, 11)
        clahe_img_lab = cv2.cvtColor(blur_img, cv2.COLOR_BGR2LAB)
        clahe_img_hsv = cv2.cvtColor(blur_img, cv2.COLOR_BGR2HSV)

        while True:
            lower_lab, upper_lab, lower_hsv, upper_hsv = get_trackbar_values(
                window_name
            )

            # Masks calculation
            lab_mask = cv2.inRange(clahe_img_lab, lower_lab, upper_lab)
            hsv_mask = cv2.inRange(clahe_img_hsv, lower_hsv, upper_hsv)

            result_mask_raw = cv2.bitwise_and(lab_mask, hsv_mask)

            # Apply morphology on the result mask
            mask_closed = cv2.morphologyEx(
                result_mask_raw, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2
            )
            result_mask = cv2.morphologyEx(
                mask_closed, cv2.MORPH_DILATE, np.ones((5, 5), np.uint8), iterations=3
            )

            res = clahe_img.copy()

            overlay = res.copy()
            overlay[result_mask == 255] = (255, 0, 255)
            res = cv2.addWeighted(overlay, 0.9, res, 0.1, 0)

            # Draw the recursive bounding boxes with a dynamic thickness
            draw_recursive_bboxes(result_mask, res)

            stacked = create_display_grid(
                original_img,
                blur_img,
                hsv_mask,
                lab_mask,
                result_mask,
                res,
                current_idx,
                len(images),
            )

            cv2.imshow(window_name, stacked)

            k = cv2.waitKey(1) & 0xFF
            if k == ord("q") or k == 27:  # q or ESC
                current_idx = len(images)  # exit outer loop
                break
            elif k == ord("n"):  # next
                current_idx += 1
                break
            elif k == ord("p"):  # prev
                current_idx = max(0, current_idx - 1)
                break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
