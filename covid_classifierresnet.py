import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import numpy as np
from tqdm import tqdm
from PIL import Image

# Define image parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 16
NUM_EPOCHS = 35
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create directories for saving models and plots
save_dir = 'model_checkpoints'
plot_dir = 'plots'
for directory in [save_dir, plot_dir]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# Define transforms for data augmentation
train_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2),
    transforms.RandomAffine(0, translate=(0.2, 0.2), scale=(0.8, 1.2)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Define ResNet model
class ResNetClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super(ResNetClassifier, self).__init__()
        self.resnet = models.resnet50(weights='IMAGENET1K_V1')  # Updated from pretrained=True
        # Freeze early layers
        for param in list(self.resnet.parameters())[:-20]:
            param.requires_grad = False
        # Replace classifier
        self.resnet.fc = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.resnet(x)

def verify_environment():
    # Check CUDA availability
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Check dataset
    if not os.path.exists(dataset_path):
        raise ValueError(f"Dataset not found at {dataset_path}")
    
    # Print dataset info
    try:
        print(f"Total images: {len(dataset)}")
        print(f"Training images: {len(train_dataset)}")
        print(f"Validation images: {len(val_dataset)}")
        print(f"Number of classes: {len(dataset.classes)}")
        print(f"Classes: {dataset.classes}")
    except Exception as e:
        print(f"Error printing dataset info: {e}")

# Check and load dataset
dataset_path = './COVID-19_Radiography_Dataset'
if not os.path.exists(dataset_path):
    raise ValueError(f"Dataset path {dataset_path} does not exist!")

try:
    dataset = datasets.ImageFolder(dataset_path, transform=train_transform)
    if len(dataset) == 0:
        raise ValueError("No images found in the dataset directory")
except Exception as e:
    raise Exception(f"Error loading dataset: {e}")

# Split into train and validation
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
generator = torch.Generator().manual_seed(42)  # For reproducibility
train_dataset, val_dataset = torch.utils.data.random_split(dataset, 
                                                         [train_size, val_size],
                                                         generator=generator)

# Create data loaders with error handling
try:
    train_loader = DataLoader(train_dataset, 
                            batch_size=BATCH_SIZE, 
                            shuffle=True, 
                            num_workers=0,
                            pin_memory=True if torch.cuda.is_available() else False)
    val_loader = DataLoader(val_dataset, 
                          batch_size=BATCH_SIZE, 
                          num_workers=0,
                          pin_memory=True if torch.cuda.is_available() else False)
except Exception as e:
    raise Exception(f"Error creating data loaders: {e}")

# Initialize model, criterion, and optimizer
try:
    model = ResNetClassifier().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.0001, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
except Exception as e:
    raise Exception(f"Error initializing model components: {e}")

# Verify environment before training
verify_environment()

# Initialize history trackers
train_acc_history = []
val_acc_history = []
train_loss_history = []
val_loss_history = []
best_val_loss = float('inf')

# Training loop
for epoch in range(NUM_EPOCHS):
    try:
        progress_bar = tqdm(total=len(train_loader), 
                          desc=f'Epoch {epoch + 1}/{NUM_EPOCHS}')
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(train_loader):
            try:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                progress_bar.set_postfix({
                    'loss': f'{running_loss/(i+1):.4f}',
                    'accuracy': f'{100. * correct / total:.2f}%'
                })
                progress_bar.update()
                
            except Exception as e:
                print(f"Error in training batch: {e}")
                continue
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                try:
                    inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item()
                    _, predicted = outputs.max(1)
                    total += labels.size(0)
                    correct += predicted.eq(labels).sum().item()
                except Exception as e:
                    print(f"Error in validation batch: {e}")
                    continue
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        print(f'\nEpoch [{epoch+1}/{NUM_EPOCHS}] Train Loss: {train_loss:.4f} Train Acc: {train_acc:.2f}% Val Loss: {val_loss:.4f} Val Acc: {val_acc:.2f}%')
        
        # Update scheduler
        scheduler.step()
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                }, os.path.join(save_dir, 'best_model_resnet.pth'))
            except Exception as e:
                print(f"Error saving best model: {e}")
        
        # Update history
        train_loss_history.append(train_loss)
        train_acc_history.append(train_acc)
        val_loss_history.append(val_loss)
        val_acc_history.append(val_acc)
        
        progress_bar.close()
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Error in epoch {epoch + 1}: {e}")
        try:
            progress_bar.close()
        except:
            pass
        continue

# Plot training results
try:
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_acc_history, label='Training Accuracy')
    plt.plot(val_acc_history, label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, 'training_history_resnet.png'))
    plt.close()
except Exception as e:
    print(f"Error plotting results: {e}")

# Save the final model
try:
    torch.save({
        'epoch': NUM_EPOCHS,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss_history[-1],
        'val_loss': val_loss_history[-1],
    }, os.path.join(save_dir, 'final_model_resnet.pth'))
except Exception as e:
    print(f"Error saving final model: {e}")

# Model evaluation
def evaluate_model(model, loader):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            try:
                inputs = inputs.to(DEVICE)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
            except Exception as e:
                print(f"Error in evaluation batch: {e}")
                continue
    
    try:
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=dataset.classes))
    except Exception as e:
        print(f"Error generating classification report: {e}")

# Evaluate the model
evaluate_model(model, val_loader)

# Function to predict single image
def predict_image(image_path):
    try:
        image = Image.open(image_path)
        image = val_transform(image).unsqueeze(0).to(DEVICE)
        
        model.eval()
        with torch.no_grad():
            output = model(image)
            _, predicted = output.max(1)
            return dataset.classes[predicted.item()]
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None 