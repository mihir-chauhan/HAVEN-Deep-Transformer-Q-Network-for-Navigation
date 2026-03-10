import cv2

def extract_last_frame(video_path, output_image_path):
    """
    Extracts the last frame from an MP4 video and saves it as an image.

    Args:
        video_path (str): The path to the input MP4 video file.
        output_image_path (str): The path where the last frame image will be saved.
                                  (e.g., "last_frame.jpg")
    """
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Error: Could not open video file at {video_path}")
        return

    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if total_frames == 0:
        print("Error: Video contains no frames.")
        cap.release()
        return

    # Set the video position to the last frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)

    ret, frame = cap.read()

    if ret:
        cv2.imwrite(output_image_path, frame)
        print(f"Last frame extracted and saved to {output_image_path}")
    else:
        print("Error: Could not read the last frame.")

    cap.release()

# Example usage:
video_file = "/Users/mihir/Data/IdeasLab/DTQN/fov/animation_20250911_185145_ep1.mp4"  # Replace with the path to your MP4 file
output_file = "image.jpg" # Desired output image file name

extract_last_frame(video_file, output_file)
