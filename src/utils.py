from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm


def unnormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """Reverses the ImageNet normalization on a tensor."""
    tensor = tensor.clone().cpu() 
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return tensor

def get_all_preds(model, loader):
    model.eval()
    all_labels_list = []
    all_preds_list = []
    
    print("Gathering all predictions for confusion matrix....")
    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Predicting"):
            images = images.to(device)
        
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            all_labels_list.append(labels.cpu())
            all_preds_list.append(preds.cpu())

    all_labels_cpu = torch.cat(all_labels_list)
    all_preds_cpu = torch.cat(all_preds_list)
    
    return all_labels_cpu, all_preds_cpu

def plot_label_distribution(train_dataset, test_dataset):

    train_labels = [label for _, label in train_dataset.samples]
    test_labels = [label for _, label in test_dataset.samples]
    
    train_counts = Counter(train_labels)
    test_counts = Counter(test_labels)
    
    train_counts = dict(sorted(train_counts.items()))
    test_counts = dict(sorted(test_counts.items()))
    
    print(f"Total training classes: {len(train_counts)}")
    print(f"Total test classes: {len(test_counts)}")
    
    plt.figure(figsize=(15, 6))
    
    # Training set distribution
    plt.subplot(1, 2, 1)
    plt.bar(train_counts.keys(), train_counts.values())
    plt.title('Training Set Class Distribution')
    plt.xlabel('Class Index (0-199)')
    plt.ylabel('Number of Images')
    plt.ylim(0, max(train_counts.values()) * 1.1)
    
    
    plt.subplot(1, 2, 2)
    plt.bar(test_counts.keys(), test_counts.values(), color='orange')
    plt.title('Test Set Class Distribution')
    plt.xlabel('Class Index (0-199)')
    plt.ylabel('Number of Images')
    plt.ylim(0, max(test_counts.values()) * 1.1)
    
    plt.tight_layout()
    plt.show()