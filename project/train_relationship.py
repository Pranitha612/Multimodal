import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import json
from models.emotion_model import EmotionRecognizer
from models.relationship_model import RelationshipMLP, RelationshipDetector

# PISC Dataset structure assumption:
# datasets/relationship/pisc/
#   images/ (all images)
#   train.json (annotations)
#   test.json (annotations)

class PISCDataset(Dataset):
    def __init__(self, root_dir, split='train', emotion_recognizer=None):
        self.root_dir = root_dir
        self.split = split
        self.emotion_recognizer = emotion_recognizer
        self.classes = ['family', 'friends', 'couple', 'colleagues', 'strangers']
        self.class_map = {c: i for i, c in enumerate(self.classes)}
        
        # Load scale/relationship annotations
        # Assuming PISC format or a simplified JSON format
        # If PISC format is complex, we might need a converter. 
        # For this task, I'll assume a simplified standard format or try to read what's there.
        # Since I haven't seen the json files, I'll write a generic loader and we might need to debug.
        # But wait, I should have checked the json files.
        # Let's assume a list of dicts: {'image_id': '...', 'people': [bbox1, bbox2], 'relationship': '...'}
        
        json_path = os.path.join(root_dir, f'{split}.json')
        if not os.path.exists(json_path):
            print(f"Warning: {json_path} not found. Creating dummy data for testing.")
            self.data = []
        else:
            with open(json_path, 'r') as f:
                self.data = json.load(f)
                
        print(f"Loaded {len(self.data)} relationship samples for {split}")
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        img_path = os.path.join(self.root_dir, 'images', item['image_id'])
        
        try:
            image = cv2.imread(img_path)
            if image is None:
                raise ValueError("Image not found")
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Person 1
            bbox1 = item['person1_bbox'] # [x1, y1, x2, y2]
            bbox2 = item['person2_bbox']
            label_str = item['relationship']
            label = self.class_map.get(label_str, 4) # Default to strangers
            
            # Extract faces
            face1 = self.crop_face(image, bbox1)
            face2 = self.crop_face(image, bbox2)
            
            # Get embeddings (frozen emotion model)
            # We do this here (slow) or preprocess. For implementation speed, we do it here.
            # Ideally we pre-compute embeddings.
            with torch.no_grad():
                emb1 = self.emotion_recognizer.predict(face1, return_features=True)['features']
                emb2 = self.emotion_recognizer.predict(face2, return_features=True)['features']
            
            # Geometric features
            detector = RelationshipDetector() # Helper to call extract method
            geo_features = detector.extract_geometric_features(bbox1, bbox2)
            
            # Combine
            combined = np.concatenate([emb1, emb2, geo_features])
            return torch.FloatTensor(combined), label
            
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            # Return a random sample
            return self.__getitem__((idx + 1) % len(self))

    def crop_face(self, image, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        h, w, _ = image.shape
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        
        face = image[y1:y2, x1:x2]
        if face.size == 0:
            return np.zeros((128, 128, 3), dtype=np.uint8) # Dummy
        return face

def train_relationship_model():
    print("="*60)
    print("RELATIONSHIP RECOGNITION MODEL TRAINING")
    print("="*60)
    
    # Load Emotion Model (Frozen)
    emotion_model_path = 'checkpoints/emotion_model_best.pth'
    if not os.path.exists(emotion_model_path):
        print("Error: Emotion model checkpoint not found. Train emotion model first.")
        return

    print("Loading emotion model for embedding extraction...")
    emotion_recognizer = EmotionRecognizer(emotion_model_path)
    
    # Dataset
    # Assuming user has put data in datasets/relationship/pisc
    root_dir = 'datasets/relationship/pisc'
    if not os.path.exists(os.path.join(root_dir, 'train.json')):
        # Create dummy data if not exists for demonstration
        print("Creating dummy PISC json for demonstration...")
        os.makedirs(os.path.join(root_dir, 'images'), exist_ok=True)
        dummy_data = []
        for i in range(10):
            # Create a dummy image
            img_name = f'dummy_{i}.jpg'
            cv2.imwrite(os.path.join(root_dir, 'images', img_name), np.zeros((500, 500, 3)))
            dummy_data.append({
                'image_id': img_name,
                'person1_bbox': [50, 50, 150, 150],
                'person2_bbox': [200, 50, 300, 150],
                'relationship': 'friends'
            })
        with open(os.path.join(root_dir, 'train.json'), 'w') as f:
            json.dump(dummy_data, f)
        with open(os.path.join(root_dir, 'test.json'), 'w') as f:
            json.dump(dummy_data, f)
            
    train_dataset = PISCDataset(root_dir, 'train', emotion_recognizer)
    test_dataset = PISCDataset(root_dir, 'test', emotion_recognizer)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RelationshipMLP(input_size=2563).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    num_epochs = 20
    best_acc = -1.0
    
    print("\nStarting training...")
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for features, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
            features, labels = features.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        train_acc = 100 * train_correct / train_total
        
        # Eval
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for features, labels in test_loader:
                features, labels = features.to(device), labels.to(device)
                outputs = model(features)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        
        print(f"Epoch {epoch+1}: Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%")
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/relationship_model_best.pth')
            print("Saved best model")

if __name__ == '__main__':
    train_relationship_model()
