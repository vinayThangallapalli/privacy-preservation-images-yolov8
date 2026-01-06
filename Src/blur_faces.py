import cv2
from ultralytics import YOLO

def blur_humans(input_video: str, output_video: str,
                model_path: str = "yolov8l.pt", frame_skip: int = 1):
    """
    Detects humans in a video using YOLOv8 and applies Gaussian blur
    to each detected human region. Saves the blurred video to output path.
    """
    # Load YOLOv8 model (trained on COCO dataset)
    model = YOLO(model_path)

    # Open input video
    cap = cv2.VideoCapture(input_video)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Prepare video writer for output
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video, fourcc, fps, (w, h))

    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process only selected frames to speed up
        if frame_idx % frame_skip == 0:
            # Run YOLO detections
            results = model(frame, verbose=False)[0]

            # Loop through detected objects
            for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
                if int(cls) == 0:  # COCO class 0 = "person"
                    x1, y1, x2, y2 = map(int, box)
                    human = frame[y1:y2, x1:x2]
                    if human.size > 0:
                        # Apply blur on human region
                        human = cv2.GaussianBlur(human, (51, 51), 30)
                        frame[y1:y2, x1:x2] = human

        # Write processed frame to output
        out.write(frame)
        frame_idx += 1

    # Release resources
    cap.release()
    out.release()
    print(f"Blurred human video saved to {output_video}")


if __name__ == "__main__":
    blur_humans("input.mp4", "blurred_humans2.mp4")
