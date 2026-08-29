# !pip install medmnist thop timm -q

import medmnist
from medmnist import PathMNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
import timm
import torch
import torch.nn as nn
import time
from thop import profile

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
assert device.type == 'cuda', "GPU not active — enable it in Runtime settings before continuing"

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_full = PathMNIST(split='train', transform=train_transform, download=True, size=28)
val_full = PathMNIST(split='val', transform=eval_transform, download=True, size=28)
test_full = PathMNIST(split='test', transform=eval_transform, download=True, size=28)


def stratified_subsample(labels, target_total, num_classes, seed=42):
    rng = np.random.RandomState(seed)
    per_class = target_total // num_classes
    indices = []
    for c in range(num_classes):
        class_idx = np.where(labels == c)[0]
        n_take = min(per_class, len(class_idx))
        chosen = rng.choice(class_idx, size=n_take, replace=False)
        indices.extend(chosen.tolist())
    rng.shuffle(indices)
    return indices


num_classes = 9

train_labels = train_full.labels.squeeze()
train_idx = stratified_subsample(train_labels, 14000, num_classes)
train_dataset = Subset(train_full, train_idx)

val_labels = val_full.labels.squeeze()
val_idx = stratified_subsample(val_labels, 2000, num_classes)
val_dataset = Subset(val_full, val_idx)

test_labels = test_full.labels.squeeze()
test_idx = stratified_subsample(test_labels, 3000, num_classes)
test_dataset = Subset(test_full, test_idx)

print("Subsampled train size:", len(train_dataset))
print("Subsampled val size:", len(val_dataset))
print("Subsampled test size:", len(test_dataset))

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

model = timm.create_model('vit_base_patch32_224', pretrained=True, num_classes=num_classes)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)


def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device).long().squeeze(1)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total


num_epochs = 20
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
train_start = time.time()
best_val_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for imgs, labels in train_loader:
        imgs = imgs.to(device)
        labels = labels.to(device).long().squeeze(1)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    scheduler.step()
    val_acc = evaluate(model, val_loader)
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch {epoch+1}/{num_epochs} - loss: {running_loss/len(train_loader):.4f} - val_acc: {val_acc:.4f} - lr: {current_lr:.6f}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_vit_base32_path.pth')

total_train_time = time.time() - train_start
print("Total training time (s):", total_train_time)
print("Best val accuracy during training:", best_val_acc)

model.load_state_dict(torch.load('best_vit_base32_path.pth'))
model.eval()

test_acc = evaluate(model, test_loader)

single_input = torch.randn(1, 3, 224, 224).to(device)
with torch.no_grad():
    for _ in range(10):
        _ = model(single_input)
start = time.time()
with torch.no_grad():
    for _ in range(50):
        _ = model(single_input)
inference_time_ms = (time.time() - start) / 50 * 1000

flops, params = profile(model, inputs=(torch.randn(1, 3, 224, 224).to(device),), verbose=False)
params_m = params / 1e6
flops_g = flops / 1e9

er = test_acc * 100 / (inference_time_ms + params_m)

print("Test Accuracy (%):", test_acc * 100)
print("Training Time (s):", total_train_time)
print("Inference Time (ms):", inference_time_ms)
print("Params (M):", params_m)
print("FLOPs (G):", flops_g)
print("Efficiency Ratio:", er)
