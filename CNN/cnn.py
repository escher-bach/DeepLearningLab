#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('nvidia-smi')


# # CNN Experiments: Cats vs Dogs and CIFAR-10
# 
# This notebook implements a configurable CNN for image classification and explores:
# - 3 activation functions (ReLU, Tanh, Leaky ReLU)
# - 3 weight initialization techniques (Xavier, Kaiming, Random)
# - 3 optimizers (SGD, Adam, RMSprop)
# 
# It then compares the best CNN for each dataset with a pretrained ResNet-18.

# In[30]:


import os
import glob
import random
import shutil
import zipfile
from dataclasses import dataclass
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models


# In[3]:


# Reproducibility and device
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device


# ## Data Preprocessing
# 
# This section prepares the datasets and dataloaders for Cats vs Dogs and CIFAR-10.

# In[31]:


KAGGLE_INPUT_ROOT = "/kaggle/input"
KAGGLE_WORKING_ROOT = "/kaggle/working"

def find_cats_dogs_root():
    candidates = glob.glob(os.path.join(KAGGLE_INPUT_ROOT, "competitions", "*"))
    for c in candidates:
        name = os.path.basename(c).lower()
        if "cat" in name and "dog" in name:
            return c
    return None

cats_dogs_input_root = find_cats_dogs_root()
cats_dogs_input_root


# In[25]:


if cats_dogs_input_root is None:
    print("Cats vs Dogs dataset not auto-detected.")
    print("Available datasets:", os.listdir(KAGGLE_INPUT_ROOT+"/competitions"))


# In[22]:


print("Available datasets:", os.listdir(KAGGLE_INPUT_ROOT+"/competitions/dogs-vs-cats"))


# In[32]:


# Transforms
cifar_mean = (0.4914, 0.4822, 0.4465)
cifar_std = (0.2470, 0.2435, 0.2616)
imagenet_mean = (0.485, 0.456, 0.406)
imagenet_std = (0.229, 0.224, 0.225)

train_transform_cifar = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std),
])
val_transform_cifar = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(cifar_mean, cifar_std),
])

train_transform_cd = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])
val_transform_cd = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(imagenet_mean, imagenet_std),
])

def _extract_zip(zip_path, extract_dir):
    if not os.path.exists(extract_dir):
        os.makedirs(extract_dir, exist_ok=True)
    sentinel = os.path.join(extract_dir, ".extracted")
    if os.path.exists(sentinel):
        return
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Zip not found: {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)
    with open(sentinel, "w") as f:
        f.write("ok")

def _dir_has_images(dir_path):
    return len(glob.glob(os.path.join(dir_path, "*.jpg"))) > 0

def prepare_cats_dogs_dataset(input_root, output_root, val_ratio=0.2, seed=SEED):
    if input_root is None:
        raise ValueError("Cats vs Dogs dataset not found. Set cats_dogs_input_root manually.")
    train_out = os.path.join(output_root, "train")
    val_out = os.path.join(output_root, "val")
    cats_train = os.path.join(train_out, "cats")
    dogs_train = os.path.join(train_out, "dogs")
    cats_val = os.path.join(val_out, "cats")
    dogs_val = os.path.join(val_out, "dogs")
    if all(os.path.isdir(p) and _dir_has_images(p) for p in [cats_train, dogs_train, cats_val, dogs_val]):
        return output_root
    if os.path.exists(train_out):
        shutil.rmtree(train_out)
    if os.path.exists(val_out):
        shutil.rmtree(val_out)
    os.makedirs(cats_train, exist_ok=True)
    os.makedirs(dogs_train, exist_ok=True)
    os.makedirs(cats_val, exist_ok=True)
    os.makedirs(dogs_val, exist_ok=True)

    # Kaggle dogs-vs-cats uses train.zip with cat/dog filenames inside
    train_zip = os.path.join(input_root, "train.zip")
    extract_root = os.path.join(KAGGLE_WORKING_ROOT, "cats_dogs_raw")
    train_extract = os.path.join(extract_root, "train")
    if os.path.exists(train_zip):
        _extract_zip(train_zip, extract_root)
    image_candidates = []
    if os.path.exists(train_extract):
        image_candidates = glob.glob(os.path.join(train_extract, "**", "*.jpg"), recursive=True)
    if len(image_candidates) == 0 and os.path.exists(extract_root):
        image_candidates = glob.glob(os.path.join(extract_root, "**", "*.jpg"), recursive=True)
    if len(image_candidates) == 0:
        image_candidates = glob.glob(os.path.join(input_root, "**", "*.jpg"), recursive=True)
    image_candidates = [p for p in image_candidates if os.path.basename(p).lower().startswith(("cat", "dog"))]
    if len(image_candidates) == 0:
        raise FileNotFoundError("No cat/dog images found under input root.")
    random.Random(seed).shuffle(image_candidates)
    split_idx = int(len(image_candidates) * (1 - val_ratio))
    train_imgs = image_candidates[:split_idx]
    val_imgs = image_candidates[split_idx:]
    for p in train_imgs:
        name = os.path.basename(p).lower()
        if name.startswith("cat"):
            shutil.copy2(p, os.path.join(cats_train, os.path.basename(p)))
        else:
            shutil.copy2(p, os.path.join(dogs_train, os.path.basename(p)))
    for p in val_imgs:
        name = os.path.basename(p).lower()
        if name.startswith("cat"):
            shutil.copy2(p, os.path.join(cats_val, os.path.basename(p)))
        else:
            shutil.copy2(p, os.path.join(dogs_val, os.path.basename(p)))
    return output_root

def get_dataloaders(dataset_name, batch_size=64, val_ratio=0.2):
    if dataset_name == "cifar10":
        train_ds = datasets.CIFAR10(root=KAGGLE_WORKING_ROOT, train=True, download=True, transform=train_transform_cifar)
        val_ds = datasets.CIFAR10(root=KAGGLE_WORKING_ROOT, train=False, download=True, transform=val_transform_cifar)
        num_classes = 10
    elif dataset_name == "cats_dogs":
        prepared_root = prepare_cats_dogs_dataset(cats_dogs_input_root, os.path.join(KAGGLE_WORKING_ROOT, "cats_dogs"), val_ratio=val_ratio)
        train_ds = datasets.ImageFolder(os.path.join(prepared_root, "train"), transform=train_transform_cd)
        val_ds = datasets.ImageFolder(os.path.join(prepared_root, "val"), transform=val_transform_cd)
        num_classes = 2
    else:
        raise ValueError("Unsupported dataset")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, num_classes


# ## Model Definition
# 
# Configurable CNN with activation, weight initialization, batch normalization, and dropout.

# In[17]:


def get_activation(name):
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    if name == "leaky_relu":
        return nn.LeakyReLU(0.1, inplace=True)
    raise ValueError("Unknown activation")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes, activation="relu"):
        super().__init__()
        act = get_activation(activation)
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            act,
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            act,
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            act,
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(128, 256),
            act,
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def init_weights(model, init_type="xavier"):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            if init_type == "xavier":
                nn.init.xavier_uniform_(m.weight)
            elif init_type == "kaiming":
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
            elif init_type == "random":
                nn.init.uniform_(m.weight, -0.05, 0.05)
            else:
                raise ValueError("Unknown init type")
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# ## Training and Evaluation
# 
# Runs all combinations of activation, initialization, and optimizer for each dataset.

# In[18]:


@dataclass
class TrainConfig:
    epochs: int = 3
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4

def make_optimizer(name, params, lr, weight_decay):
    if name == "sgd":
        return optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    if name == "adam":
        return optim.Adam(params, lr=lr, weight_decay=weight_decay)
    if name == "rmsprop":
        return optim.RMSprop(params, lr=lr, weight_decay=weight_decay)
    raise ValueError("Unknown optimizer")

def run_epoch(model, loader, criterion, optimizer=None):
    running_loss = 0.0
    running_correct = 0
    total = 0
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        if is_train:
            optimizer.zero_grad()
        with torch.set_grad_enabled(is_train):
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            if is_train:
                loss.backward()
                optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        running_correct += torch.sum(preds == labels).item()
        total += labels.size(0)
    avg_loss = running_loss / total
    avg_acc = running_correct / total
    return avg_loss, avg_acc

def train_model(model, train_loader, val_loader, optimizer, epochs=3):
    criterion = nn.CrossEntropyLoss()
    best_acc = -1.0
    best_state = None
    history = []
    for epoch in range(epochs):
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    return best_state, best_acc, history

def run_grid_search(dataset_name, config: TrainConfig):
    activations = ["relu", "tanh", "leaky_relu"]
    inits = ["xavier", "kaiming", "random"]
    optimizers = ["sgd", "adam", "rmsprop"]
    train_loader, val_loader, num_classes = get_dataloaders(dataset_name, batch_size=config.batch_size)
    results = []
    best = {"acc": -1.0, "state": None, "meta": None}
    for act in activations:
        for init in inits:
            for opt_name in optimizers:
                model = SimpleCNN(num_classes=num_classes, activation=act).to(device)
                init_weights(model, init_type=init)
                optimizer = make_optimizer(opt_name, model.parameters(), config.lr, config.weight_decay)
                best_state, best_acc, history = train_model(model, train_loader, val_loader, optimizer, epochs=config.epochs)
                print(
                    f"[{dataset_name}] activation={act}, init={init}, optimizer={opt_name} -> best_val_acc={best_acc:.4f}"
                )
                results.append({
                    "dataset": dataset_name,
                    "activation": act,
                    "init": init,
                    "optimizer": opt_name,
                    "best_val_acc": best_acc,
                })
                if best_acc > best["acc"]:
                    best = {
                        "acc": best_acc,
                        "state": best_state,
                        "meta": {"activation": act, "init": init, "optimizer": opt_name},
                    }
    return results, best, num_classes

def train_best_model(dataset_name, best_meta, num_classes, config: TrainConfig, epochs=10):
    train_loader, val_loader, _ = get_dataloaders(dataset_name, batch_size=config.batch_size)
    model = SimpleCNN(num_classes=num_classes, activation=best_meta["activation"]).to(device)
    init_weights(model, init_type=best_meta["init"])
    optimizer = make_optimizer(best_meta["optimizer"], model.parameters(), config.lr, config.weight_decay)
    best_state, best_acc, history = train_model(model, train_loader, val_loader, optimizer, epochs=epochs)
    print(
        f"[{dataset_name}] final training -> activation={best_meta['activation']}, init={best_meta['init']}, optimizer={best_meta['optimizer']} | best_val_acc={best_acc:.4f}"
    )
    return best_state, best_acc, history


# In[19]:


config = TrainConfig(epochs=3, batch_size=64, lr=1e-3, weight_decay=1e-4)

# CIFAR-10 grid search
cifar_results, cifar_best, cifar_num_classes = run_grid_search("cifar10", config)
cifar_results_df = pd.DataFrame(cifar_results)
cifar_results_df.to_csv(os.path.join(KAGGLE_WORKING_ROOT, "cifar10_cnn_results.csv"), index=False)
cifar_results_df.sort_values("best_val_acc", ascending=False).head()

# Train best CIFAR-10 model longer
cifar_final_state, cifar_final_acc, _ = train_best_model(
    "cifar10",
    cifar_best["meta"],
    cifar_num_classes,
    config,
    epochs=10,
)
cifar_final = {"state": cifar_final_state, "acc": cifar_final_acc, "meta": cifar_best["meta"]}


# Best performing is Leaky Relu Xavier Adam in this set of runs with

# In[33]:


# Cats vs Dogs grid search
catsdogs_results, catsdogs_best, catsdogs_num_classes = run_grid_search("cats_dogs", config)
catsdogs_results_df = pd.DataFrame(catsdogs_results)
catsdogs_results_df.to_csv(os.path.join(KAGGLE_WORKING_ROOT, "cats_dogs_cnn_results.csv"), index=False)
catsdogs_results_df.sort_values("best_val_acc", ascending=False).head()

# Train best Cats vs Dogs model longer
catsdogs_final_state, catsdogs_final_acc, _ = train_best_model(
    "cats_dogs",
    catsdogs_best["meta"],
    catsdogs_num_classes,
    config,
    epochs=10,
)
catsdogs_final = {"state": catsdogs_final_state, "acc": catsdogs_final_acc, "meta": catsdogs_best["meta"]}


# The best performing in case of Cats and Dogs was Relu Xavier and Adam

# In[ ]:


def save_best_weights(best_info, dataset_name, num_classes):
    model = SimpleCNN(num_classes=num_classes, activation=best_info["meta"]["activation"])
    model.load_state_dict(best_info["state"])
    weights_path = os.path.join(
        KAGGLE_WORKING_ROOT,
        f"{dataset_name}_best_cnn_{best_info['meta']['activation']}_{best_info['meta']['init']}_{best_info['meta']['optimizer']}.pth"
    )
    torch.save(model.state_dict(), weights_path)
    return weights_path

cifar_best_path = save_best_weights(cifar_final, "cifar10", cifar_num_classes)
catsdogs_best_path = save_best_weights(catsdogs_final, "cats_dogs", catsdogs_num_classes)
cifar_best_path, catsdogs_best_path


# ## Learning with ResNet-18
# 
# Fine-tune a pretrained ResNet-18 and compare against the best CNN results.

# In[34]:


def get_resnet_model(num_classes):
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def train_resnet18(dataset_name, num_classes, epochs=3, lr=1e-4, weight_decay=1e-4):
    train_loader, val_loader, _ = get_dataloaders(dataset_name, batch_size=64)
    model = get_resnet_model(num_classes)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    best_state, best_acc, history = train_model(model, train_loader, val_loader, optimizer, epochs=epochs)
    return best_state, best_acc, history

resnet_cifar_state, resnet_cifar_acc, _ = train_resnet18("cifar10", cifar_num_classes, epochs=3)
resnet_catsdogs_state, resnet_catsdogs_acc, _ = train_resnet18("cats_dogs", catsdogs_num_classes, epochs=3)

resnet_cifar_path = os.path.join(KAGGLE_WORKING_ROOT, "cifar10_resnet18_finetuned.pth")
resnet_catsdogs_path = os.path.join(KAGGLE_WORKING_ROOT, "cats_dogs_resnet18_finetuned.pth")
torch.save(resnet_cifar_state, resnet_cifar_path)
torch.save(resnet_catsdogs_state, resnet_catsdogs_path)
resnet_cifar_acc, resnet_catsdogs_acc


# ## Results Summary
# 
# Compare the best CNN results with ResNet-18 for each dataset.

# In[35]:


summary = pd.DataFrame([
    {
        "dataset": "cifar10",
        "best_cnn_acc": cifar_final["acc"],
        "best_cnn_activation": cifar_final["meta"]["activation"],
        "best_cnn_init": cifar_final["meta"]["init"],
        "best_cnn_optimizer": cifar_final["meta"]["optimizer"],
        "resnet18_acc": resnet_cifar_acc,
    },
    {
        "dataset": "cats_dogs",
        "best_cnn_acc": catsdogs_final["acc"],
        "best_cnn_activation": catsdogs_final["meta"]["activation"],
        "best_cnn_init": catsdogs_final["meta"]["init"],
        "best_cnn_optimizer": catsdogs_final["meta"]["optimizer"],
        "resnet18_acc": resnet_catsdogs_acc,
    },
])
summary.to_csv(os.path.join(KAGGLE_WORKING_ROOT, "summary_results.csv"), index=False)
summary

