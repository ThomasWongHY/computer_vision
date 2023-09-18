# computer_vision

## object_detection
In this project, we will use YOLO v8 from ultralytics for object detection. 

We process a dogs video as a sequence of images by OpenCV to apply object detection with Yolo v8.
```python
cap = cv2.VideoCapture("dogs.mp4")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    cv2.imshow("Img", frame)
    key = cv2.waitKey(1)
    if key == 27:  # Press "esc" key to exit
        break
```
Then we can draw rectangles on the dogs by obtaining the coordinates of the bounding box related to the two dogs in the video.
```python
results = model(frame, device="mps")
result = results[0]
bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
classes = np.array(result.boxes.cls.cpu(), dtype="int")

for cls, bbox in zip(classes, bboxes):
    (x, y, x2, y2) = bbox
    
    cv2.rectangle(frame, (x, y), (x2, y2), (0, 0, 255), 2)
    cv2.putText(frame, str(cls), (x, y - 5), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 225), 2)
```

<img width="1512" alt="Screenshot 2023-09-17 at 23 04 35" src="https://github.com/ThomasWongHY/computer_vision/assets/86035047/e2c58959-f882-4fb5-9ede-e8a05aa4f364">

## monitor_time_tracker.py
For this exercise, we will track the time in which users are in front of the computer by automatically resetting it to zero when the software detects that the user has moved away by the OpenCV library and python.

First, we can set the maximum time for users to show alert and the timer when the app opens.
```python
# Settings
maximum_time = 10

# Load Face Detector
face_detection = mp.solutions.face_detection.FaceDetection() 

# Take frame from capera
cap = cv2.VideoCapture(0)

# Track Time
starting_time = time.time()
```
Then we draw the rectangle around the users' face by face detection if it detects face in front of the webcam. When the elapsed time is larger than the maximum time, it will show an alert which is a red rectangle and the cv2 window poping up to the top most in this case.
However, it will reset the timer if users are away from keyboard.
```python
while True:
    # Take frame from camera
    ret, frame = cap.read()
    height, width, channels = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Draw rectangle
    cv2.rectangle(frame, (0, 0), (width, 70), (10, 10, 10), -1)
    
    # Face Detection
    results = face_detection.process(rgb_frame)
    
    # Is the face detected?
    if results.detections:
        elapsed_time = int(time.time() - starting_time)
        
        if elapsed_time > maximum_time:
            # Reached maximum time, show alert
            cv2.rectangle(frame, (0, 0), (width, height), (0, 0, 225), 10)
            cv2.setWindowProperty("Frame", cv2.WND_PROP_TOPMOST, 1)
            
        # Draw elapsed time on screen
        cv2.putText(frame, f"{elapsed_time} seconds", (10, 50), cv2.FONT_HERSHEY_PLAIN, 3, (15, 225, 215), 2)
        print(f"Elapsed: {elapsed_time}")
        print("Detected")
    else:
        print('No Face')
        # Reset the counter
        starting_time = time.time()
```

## Instance Segmentation
In this project, we will utilize computer vision to apply segmentation to objects with Yolov8 by Ultralitycs. With the segmentation, the object’s shape is identified, allowing the calculation of its size.
First, we have a yolo segmentation class to obtain the image results and return the bounding boxes, class ids, segmentation contours indice and scores.
```python
for seg in result.masks.segments:
    # contours
    seg[:, 0] *= width
    seg[:, 1] *= height
    segment = np.array(seg, dtype=np.int32)
    segmentation_contours_idx.append(segment)

bboxes = np.array(result.boxes.xyxy.cpu(), dtype="int")
# Get class ids
class_ids = np.array(result.boxes.cls.cpu(), dtype="int")
# Get scores
scores = np.array(result.boxes.conf.cpu(), dtype="float").round(2)
return bboxes, class_ids, segmentation_contours_idx, scores
```

Then we utilize these returned values to draw rectangles and polylines on the target instance as well as show the class id.
```python
for bbox, class_id, seg, score in zip(bboxes, classes, segmentation, scores):
    # print("bbox: ", bbox, "class id: ", class_id, "seg: ", seg, "score: ", score)
    
    (x, y, x2, y2) = bbox
    cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 2)
    
    cv2.polylines(img, [seg], True, (255, 0, 0), 2)
    
    cv2.putText(img, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
```
<img width="1512" alt="Screenshot 2023-09-17 at 23 30 21" src="https://github.com/ThomasWongHY/computer_vision/assets/86035047/ad41d546-8e86-4b34-9591-44e7a87f98e6">
