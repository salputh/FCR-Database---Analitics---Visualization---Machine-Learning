import cv2
import numpy as np
import sys

print(sys.executable)

print(cv2.__version__)


def main():
    # Initialize the built-in webcam (index 0)
    cap = cv2.VideoCapture(0, cv2.CAP_ANY)  # Use CAP_ANY for macOS compatibility

    if not cap.isOpened():
        print("Failed to open the built-in webcam.")
        return

    # Set window name
    window_name = "Built-in Webcam Feed"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame from the webcam.")
            break

        # Display the webcam feed
        cv2.imshow(window_name, frame)

        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
