import cv2
from yolo_segmentation import YOLOSegmentation

img = cv2.imread("/Users/thomas/Downloads/ComputerVision/instance_segmentation/images/football.jpg")
img = cv2.resize(img, None, fx=0.7, fy=0.7)

# Segementation detector
ys = YOLOSegmentation("yolov8m-seg.pt")

bboxes, classes, segmentation, scores = ys.detect(img)
for bbox, class_id, seg, score in zip(bboxes, classes, segmentation, scores):
    # print("bbox: ", bbox, "class id: ", class_id, "seg: ", seg, "score: ", score)
    
    (x, y, x2, y2) = bbox
    cv2.rectangle(img, (x, y), (x2, y2), (0, 0, 255), 2)
    
    cv2.polylines(img, [seg], True, (255, 0, 0), 2)
    
    cv2.putText(img, str(class_id), (x, y - 10), cv2.FONT_HERSHEY_PLAIN, 5, (0, 0, 255), 5)
    
    
cv2.imshow("image", img)
cv2.waitKey(0)