import cv2
from djitellopy import Tello
import time

def main():
    # Initialize the Tello drone
    maverick = Tello()

    # Connect to the drone
    print("Connecting to Tello drone...")
    maverick.connect()
    print("Connected to Tello drone.")

    # Check battery level
    battery_level = maverick.get_battery()
    print(f"Battery level: {battery_level}%")
    if battery_level <= 20:
        print("Battery level is too low. Please charge the drone.")
        return

    # Start the video stream
    print("Starting video stream...")
    maverick.streamon()
    time.sleep(1)  # Allow some time for the stream to start
    print("Video stream started.")

    # Create a window to display the video feed
    cv2.namedWindow("Tello Video Feed", cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            # Get a frame from the Tello video stream
            frame = maverick.get_frame_read().frame

            if frame is None:
                print("Failed to get frame from Tello video stream.")
                continue

            # Display the frame
            cv2.imshow("Tello Video Feed", frame)

            # Exit the loop when 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Clean up
        print("Cleaning up...")
        cv2.destroyAllWindows()
        maverick.streamoff()
        maverick.end()
        print("Cleanup complete.")

if __name__ == "__main__":
    main()