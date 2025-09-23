import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.models import resnet18, ResNet18_Weights
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.metrics import f1_score, roc_auc_score
import cv2
import numpy as np
import os

# ----------------------------
# 1️⃣ Config
# ----------------------------
train_dir = "cls_dataset_binary/train"
val_dir = "cls_dataset_binary/val"
batch_size = 32
num_epochs = 20
lr = 0.001
device = torch.device("cpu")  # ✅ Force CPU
model_save_path = "best_classification_model.pth"

print(f"Using device: {device}")

# ----------------------------
# 2️⃣ Albumentations transforms
# ----------------------------
train_transform = A.Compose([
    A.Resize(224, 224),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.ShiftScaleRotate(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ----------------------------
# 3️⃣ Custom dataset
# ----------------------------
class AlbumentationsDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.samples = []
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        for label_idx, cls in enumerate(self.classes):
            cls_dir = os.path.join(root_dir, cls)
            for f in os.listdir(cls_dir):
                if f.endswith(".png") or f.endswith(".jpg"):
                    self.samples.append((os.path.join(cls_dir, f), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image=image)["image"]
        # ✅ Ensure labels are torch.long (int64)
        return image, torch.tensor(label, dtype=torch.long)

train_dataset = AlbumentationsDataset(train_dir, transform=train_transform)
val_dataset = AlbumentationsDataset(val_dir, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

num_classes = len(train_dataset.classes)
print(f"Classes: {train_dataset.classes}, Number of classes: {num_classes}")
print(f"Number of training samples: {len(train_dataset)}")
print(f"Number of validation samples: {len(val_dataset)}")

# 🔍 Debug check for one batch
sample_images, sample_labels = next(iter(train_loader))
print("DEBUG batch:", sample_images.dtype, sample_images.min().item(), sample_images.max().item(), sample_images.shape)
print("DEBUG labels dtype:", sample_labels.dtype)

# ----------------------------
# 4️⃣ Model
# ----------------------------
model = resnet18(weights=ResNet18_Weights.DEFAULT)
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)

if os.path.exists(model_save_path):
    model.load_state_dict(torch.load(model_save_path, map_location=device))
    print(f"Loaded {model_save_path}, continuing training...")

model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# ----------------------------
# 5️⃣ Training & Validation Loop with Early Stopping
# ----------------------------
best_val_f1 = 0.0
best_model_wts = model.state_dict()  # keep best weights in memory
patience = 5
epochs_no_improve = 0

for epoch in range(num_epochs):
    print(f"\n===== Epoch [{epoch+1}/{num_epochs}] =====")
    
    # Training
    model.train()
    running_loss = 0.0
    for step, (images, labels) in enumerate(train_loader, start=1):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        
        if step % 10 == 0 or step == len(train_loader):
            print(f"[Train] Step {step}/{len(train_loader)}, Batch Loss: {loss.item():.4f}")
    
    epoch_loss = running_loss / len(train_dataset)
    print(f"[Train] Epoch Loss: {epoch_loss:.4f}")

    # Validation
    model.eval()
    all_labels, all_preds, all_probs = [], [], []
    with torch.no_grad():
        for step, (images, labels) in enumerate(val_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs[:,1].cpu().numpy())  # abnormal prob
            
            if step % 5 == 0 or step == len(val_loader):
                print(f"[Val] Step {step}/{len(val_loader)} completed")
    
    val_acc = np.mean(np.array(all_preds) == np.array(all_labels))
    val_f1 = f1_score(all_labels, all_preds, average='weighted')
    try: val_auc = roc_auc_score(all_labels, all_probs)
    except: val_auc = None
    print(f"[Val] Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}, AUC: {val_auc}")

    # ----------------------------
    # Early stopping logic
    # ----------------------------
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_wts = model.state_dict()  # keep best weights
        torch.save(best_model_wts, model_save_path)
        print(f"[Save] Best model updated with F1-score: {best_val_f1:.4f}")
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1
        print(f"[Early Stop] No improvement for {epochs_no_improve} epochs")

        if epochs_no_improve >= patience:
            print("\n⏹️ Early stopping triggered!")
            break

# ----------------------------
# 6️⃣ Restore best weights
# ----------------------------
model.load_state_dict(best_model_wts)
print("\n✅ Training Complete! Restored best model weights.")
print(f"Best Validation F1-score: {best_val_f1:.4f}")
