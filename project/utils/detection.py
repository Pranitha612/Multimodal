import cv2
from ultralytics import YOLO
from mtcnn import MTCNN
import numpy as np

class PersonDetector:
    def __init__(self):
        """Initialize YOLOv8 for person detection"""
        print("Loading YOLOv8 model (Medium)...")
        self.model = YOLO('yolov8m.pt')
        print("YOLOv8 model loaded successfully!")
        
    def detect_persons(self, image_path):
        """Detect all persons in the image"""
        # Adjusted conf to 0.55 and iou to 0.65 to allow overlapping detections (hugging)
        results = self.model(image_path, conf=0.55, iou=0.65)
        persons = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                if int(box.cls[0]) == 0:  # person class
                    coords = box.xyxy[0].cpu().numpy()
                    persons.append({
                        'bbox': coords,
                        'confidence': float(box.conf[0])
                    })
        
        return persons

class FaceDetector:
    def __init__(self):
        """Initialize MTCNN for face detection"""
        print("Loading MTCNN face detector...")
        self.detector = MTCNN()
        print("MTCNN loaded successfully!")
        
    def extract_faces(self, image, person_boxes):
        """Extract faces from person bounding boxes"""
        if isinstance(image, str):
            image = cv2.imread(image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        faces = []
        
        for person in person_boxes:
            x1, y1, x2, y2 = map(int, person['bbox'])
            person_roi = image[y1:y2, x1:x2]
            
            if person_roi.size == 0:
                continue
                
            face_data = self.detector.detect_faces(person_roi)
            
            if face_data:
                face = max(face_data, key=lambda x: x['confidence'])
                fx, fy, fw, fh = face['box']
                
                face_img = person_roi[fy:fy+fh, fx:fx+fw]
                
                faces.append({
                    'image': face_img,
                    'bbox': (x1+fx, y1+fy, x1+fx+fw, y1+fy+fh),
                    'person_bbox': person['bbox'],
                    'confidence': face['confidence']
                })
        
        return faces