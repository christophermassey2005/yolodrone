# yolodrone
I bought a TELLO drone and decided to use YOLO (you only look once) object detection whilst flying it.
As the drone only uses an android app, I had to use scrcpy to stream the images live to my computer from my phone, crop them, and then pass them to my YOLO model (which used CUDA to maintain a roughly 20 FPS live prediction stream on my RTX 4060 GPU.
Below is a snapshot of what it was able to do live.
![image](https://github.com/user-attachments/assets/7cee542a-ec12-4ffb-9991-d1864c681226)
