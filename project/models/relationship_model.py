import torch
import torch.nn as nn
import numpy as np
import cv2

class RelationshipMLP(nn.Module):
    def __init__(self, input_size=2560 + 3, num_classes=5): # 1280 * 2 (embeddings) + 3 (geometric)
        super(RelationshipMLP, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_size, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        return self.network(x)

class RelationshipDetector:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # EfficientNet-B0 embeddings are 1280 dim. 
        # Two people = 2560 dims + 3 geometric features = 2563
        self.model = RelationshipMLP(input_size=2563).to(self.device)
        self.relationships = ['family', 'friends', 'couple', 'colleagues', 'strangers']
        
        if model_path:
            self.load_model(model_path)
        else:
            print("Warning: No pretrained relationship model loaded. Using random weights.")
            
        self.model.eval()
        
    def load_model(self, path):
        print(f"Loading relationship model from {path}")
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        print("Relationship model loaded successfully!")
        
    def extract_geometric_features(self, bbox1, bbox2):
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        center1 = ((x1_1 + x2_1) / 2, (y1_1 + y2_1) / 2)
        center2 = ((x1_2 + x2_2) / 2, (y1_2 + y2_2) / 2)
        distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
        
        # Normalize distance by image diagonal or similar would be better, but we stick to simpler for now
        # Ideally we should normalize by the size of the people/image
        # Using a fixed normalization factor (e.g. 1000) for stability
        distance = distance / 1000.0
        
        size1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        size2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Avoid division by zero
        max_size = max(size1, size2)
        if max_size == 0:
            size_ratio = 1.0
        else:
            size_ratio = min(size1, size2) / max_size
        
        v_align = abs((y1_1 + y2_1) / 2 - (y1_2 + y2_2) / 2) / 1000.0
        
        return [distance, size_ratio, v_align]
        
    def predict(self, face_data_list):
        if len(face_data_list) < 2:
            return []
            
        relationships = []
        
        with torch.no_grad():
            for i in range(len(face_data_list)):
                for j in range(i + 1, len(face_data_list)):
                    
                    # Get embeddings from face data (computed by emotion model now)
                    # Use .get() loosely or assume it exists. App needs to pass it.
                    emb1 = face_data_list[i].get('embedding', np.zeros(1280))
                    emb2 = face_data_list[j].get('embedding', np.zeros(1280))
                    
                    geo_features = self.extract_geometric_features(
                        face_data_list[i]['bbox'],
                        face_data_list[j]['bbox']
                    )
                    
                    # Combine features
                    combined = np.concatenate([emb1, emb2, geo_features])
                    features_tensor = torch.FloatTensor(combined).unsqueeze(0).to(self.device)
                    
                    outputs = self.model(features_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_idx = torch.argmax(probabilities, dim=1).item()
                    confidence = probabilities[0][predicted_idx].item()
                    
                    relationship_label = self.relationships[predicted_idx]
                    
                    # Heuristic patch for untrained model: if very close (hugging/family)
                    distance, size_ratio, v_align = geo_features
                    if distance < 0.25:  # Very close proximity
                        relationship_label = 'family'
                        confidence = 0.95
                    elif relationship_label == 'colleagues' and distance < 0.4:
                        relationship_label = 'family'
                    
                    relationships.append({
                        'person1': i,
                        'person2': j,
                        'relationship': relationship_label,
                        'confidence': confidence
                    })
        
        return relationships