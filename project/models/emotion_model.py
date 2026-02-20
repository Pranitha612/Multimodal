import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
import cv2
from torchvision import transforms

class EmotionCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(EmotionCNN, self).__init__()
        
        # Use EfficientNet-B0
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        
        # EfficientNet input is usually 3 channels, but we are using grayscale
        # We can either modify the first layer or repeat the grayscale image to 3 channels
        # Modifying first layer to accept 1 channel
        first_conv_layer = self.backbone.features[0][0]
        self.backbone.features[0][0] = nn.Conv2d(
            1, first_conv_layer.out_channels, 
            kernel_size=first_conv_layer.kernel_size, 
            stride=first_conv_layer.stride, 
            padding=first_conv_layer.padding, 
            bias=False
        )
        
        # Replace classifier
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.2, inplace=True),
            nn.Linear(num_features, num_classes),
        )
        
    def forward(self, x, return_features=False):
        # Extract features
        features = self.backbone.features(x)
        features = self.backbone.avgpool(features)
        features = torch.flatten(features, 1)
        
        if return_features:
            return features
            
        output = self.backbone.classifier(features)
        return output

class EmotionRecognizer:
    def __init__(self, model_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        self.model = EmotionCNN(num_classes=7).to(self.device)
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        
        if model_path:
            self.load_model(model_path)
        else:
            print("Warning: No pretrained emotion model loaded. Using random weights.")
            
        self.model.eval()
        
    def load_model(self, path):
        print(f"Loading emotion model from {path}")
        try:
            self.model.load_state_dict(torch.load(path, map_location=self.device))
            print("Emotion model loaded successfully!")
        except Exception as e:
            print(f"Warning: Failed to load emotion model: {e}")
            print("Using random weights instead.")
        
    def preprocess(self, face_image):
        if len(face_image.shape) == 3:
            face_gray = cv2.cvtColor(face_image, cv2.COLOR_RGB2GRAY)
        else:
            face_gray = face_image
            
        # EfficientNet typically expects larger images, 224x224 is standard, 
        # but 48x48 was used for FER2013. We'll upscale to at least 128 for better features
        face_resized = cv2.resize(face_gray, (128, 128))
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        face_tensor = transform(face_resized).unsqueeze(0)
        return face_tensor.to(self.device)
        
    def predict(self, face_image, return_features=False):
        with torch.no_grad():
            face_tensor = self.preprocess(face_image)
            

            if return_features:
                features = self.model(face_tensor, return_features=True)
                outputs = self.model.backbone.classifier(features)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()
                
                return {
                    'emotion': self.emotions[predicted_idx],
                    'confidence': confidence,
                    'features': features.cpu().numpy().flatten(),
                    'all_probabilities': {
                        self.emotions[i]: probabilities[0][i].item() 
                        for i in range(len(self.emotions))
                    }
                }
            else:
                outputs = self.model(face_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                predicted_idx = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0][predicted_idx].item()
                
                return {
                    'emotion': self.emotions[predicted_idx],
                    'confidence': confidence,
                    'all_probabilities': {
                        self.emotions[i]: probabilities[0][i].item() 
                        for i in range(len(self.emotions))
                    }
                }