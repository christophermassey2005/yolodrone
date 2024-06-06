"""
import cv2
import numpy as np
from Xlib import display, X
from ultralytics import YOLO 
import supervision as sv

def get_window_by_name(window_name):
    d = display.Display()
    root = d.screen().root
    window_ids = root.get_full_property(d.intern_atom('_NET_CLIENT_LIST'), X.AnyPropertyType).value
    for window_id in window_ids:
        window = d.create_resource_object('window', window_id)
        if window.get_wm_name() == window_name:
            return window
    return None

def capture_window(window_name):
    window = get_window_by_name(window_name)
    if window is None:
        print(f"Window '{window_name}' not found.")
        return None

    geo = window.get_geometry()
    w = geo.width
    h = geo.height
    
    raw_img = window.get_image(0, 0, w, h, X.ZPixmap, 0xffffffff)
    image = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(h, w, 4)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    return image

#Set up the model
model = YOLO("yolov8l.pt")
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
)


# Specify the window name to capture
window_name = "CPH2207"

# Capture the window
image = capture_window(window_name)

if image is not None:
    # Display the captured image

    result = model(image)[0]
    detections = sv.Detections.from_ultralytics(result)
    labels = [
        f"{model.model.names[class_id]} {confidence:0.2f}"
        for _, confidence, class_id, _
        in detections
    ]
    frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)

    cv2.imshow("Captured Window", image)


    cv2.waitKey(0)
    cv2.destroyAllWindows()
"""
import cv2
import numpy as np
from Xlib import display, X
from ultralytics import YOLO 
import supervision as sv

def get_window_by_name(window_name):
    d = display.Display()
    root = d.screen().root
    window_ids = root.get_full_property(d.intern_atom('_NET_CLIENT_LIST'), X.AnyPropertyType).value
    for window_id in window_ids:
        window = d.create_resource_object('window', window_id)
        if window.get_wm_name() == window_name:
            return window
    return None

def capture_window(window_name):
    window = get_window_by_name(window_name)
    if window is None:
        print(f"Window '{window_name}' not found.")
        return None

    geo = window.get_geometry()
    w = geo.width
    h = geo.height
    
    raw_img = window.get_image(0, 0, w, h, X.ZPixmap, 0xffffffff)
    image = np.frombuffer(raw_img.data, dtype=np.uint8).reshape(h, w, 4)
    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    
    return image

# Set up the model
model = YOLO("yolov8l.pt")
box_annotator = sv.BoxAnnotator(
    thickness=2,
    text_thickness=2,
    text_scale=1
)

# Specify the window name to capture
window_name = "CPH2207"

# Capture the window
image = capture_window(window_name)

if image is not None:
    # Display the captured image
    result = model(image)[0]

    # Print the result to debug the structure
    print("Result structure:", result)

    detections = sv.Detections.from_ultralytics(result)

    # Print the detections to see their structure
    print("Detections:", detections)

    labels = []
    for i in range(len(detections.xyxy)):
        box = detections.xyxy[i]
        confidence = detections.confidence[i]
        class_id = detections.class_id[i]
        labels.append(f"{model.model.names[class_id]} {confidence:0.2f}")

    frame = box_annotator.annotate(scene=image, detections=detections, labels=labels)

    cv2.imshow("Captured Window", frame)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
