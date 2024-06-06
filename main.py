import cv2
import numpy as np
from Xlib import display, X
from ultralytics import YOLO 
import supervision as sv

import ollama
"""
modelfile = '''
FROM llama3
SYSTEM Either reply yes or no. It does not matter if it makes sense or not. Do not reply with anything else.
'''

ollama.create(model='dronepilot', modelfile=modelfile)

messages = [
    {
        'role': 'user',
        'content': 'Person detected',
    },
]

print("Response:")
for chunk in ollama.chat(model='dronepilot', messages=messages, stream=True):
    print(chunk['message']['content'], end='', flush=True)
print()

"""

ZONE_POLYGON = np.array([
    [0, 0],
    [1280, 0],
    [1250, 720],
    [0, 720]
])

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
bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=2)
label_annotator = sv.LabelAnnotator(text_scale=1)

zone = sv.PolygonZone(polygon=ZONE_POLYGON)
zone_annotator = sv.PolygonZoneAnnotator(zone=zone, color=sv.Color.RED)

# Specify the window name to capture
window_name = "CPH2207"

while True:
    # Capture the window
    image = capture_window(window_name)
    
    if image is not None:
        # Display the captured image
        result = model(image)[0]

        detections = sv.Detections.from_ultralytics(result)

        labels = []
        detected_object_names = []
        for i in range(len(detections.xyxy)):
            box = detections.xyxy[i]
            confidence = detections.confidence[i]
            class_id = detections.class_id[i]
            labels.append(f"{model.model.names[class_id]} {confidence:0.2f}")
            detected_object_names.append(model.model.names[class_id])
            
            # Print bounding box coordinates
            x1, y1, x2, y2 = box.tolist()
            print(f"Bounding box coordinates for {model.model.names[class_id]}: (x1={x1}, y1={y1}), (x2={x2}, y2={y2})")
        
        print("Detected objects:", detected_object_names)

        frame = bounding_box_annotator.annotate(scene=image, detections=detections)
        frame = label_annotator.annotate(scene=frame, detections=detections, labels=labels)

        zone.trigger(detections=detections)
        frame = zone_annotator.annotate(scene=frame)

        cv2.imshow("Captured Window", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
