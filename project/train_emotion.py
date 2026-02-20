import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
from PIL import Image
from tqdm import tqdm
from models.emotion_model import EmotionCNN

class EmotionDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.images = []
        self.labels = []
        
        for idx, emotion in enumerate(self.classes):
            emotion_dir = os.path.join(root_dir, emotion)
            if os.path.exists(emotion_dir):
                for img_name in os.listdir(emotion_dir):
                    img_path = os.path.join(emotion_dir, img_name)
                    if os.path.isfile(img_path):
                        self.images.append(img_path)
                        self.labels.append(idx)
        
        print(f"Found {len(self.images)} images in {root_dir}")
                    
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = self.images[idx]
        try:
            image = Image.open(img_path).convert('L')
            label = self.labels[idx]
            
            if self.transform:
                image = self.transform(image)
                
            return image, label
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            return self.__getitem__((idx + 1) % len(self))

def train_emotion_model():
    print("="*60)
    print("EMOTION RECOGNITION MODEL TRAINING")
    print("="*60)
    
    batch_size = 64
    num_epochs = 50
    learning_rate = 0.001
    
    train_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # Added augmentation
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])
    
    print("\nLoading datasets...")
    train_dataset = EmotionDataset('datasets/emotion/fer2013/train', transform=train_transform)
    test_dataset = EmotionDataset('datasets/emotion/fer2013/test', transform=test_transform)
    
    if len(train_dataset) == 0:
        print("ERROR: No training images found!")
        print("Please ensure images are in: datasets/emotion/fer2013/train/")
        return
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    model = EmotionCNN(num_classes=7).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3)
    
    best_acc = 0.0
    
    print("\nStarting training...")
    print("="*60)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': train_loss/len(train_loader), 
                            'acc': 100*train_correct/train_total})
        
        model.eval()
        test_loss = 0.0
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_acc = 100 * test_correct / test_total
        avg_test_loss = test_loss / len(test_loader)
        
        print(f'\nEpoch {epoch+1}: Train Acc: {100*train_correct/train_total:.2f}%, '
              f'Test Acc: {test_acc:.2f}%, Test Loss: {avg_test_loss:.4f}')
        
        scheduler.step(avg_test_loss)
        
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), 'checkpoints/emotion_model_best.pth')
            print(f'✓ Saved best model with accuracy: {best_acc:.2f}%')
    
    print("\n" + "="*60)
    print(f'Training completed! Best accuracy: {best_acc:.2f}%')
    print("="*60)

if __name__ == '__main__':
    os.makedirs('checkpoints', exist_ok=True)
    train_emotion_model()