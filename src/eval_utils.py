import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

from xai_utils import calculate_ser_captum, calculate_pointing_game

def plot_cm(true_labels, pred_labels):
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(20, 20))
    sns.heatmap(
        cm_normalized,
        annot=False,         
        cmap='viridis',      
        cbar=True,
        xticklabels=False,  
        yticklabels=False   
    )
    plt.title('Normalized Confusion Matrix (200x200)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def print_confused_classes(true_labels, pred_labels, class_names):
    cm = confusion_matrix(true_labels, pred_labels)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    df_cm = pd.DataFrame(
        cm_normalized, 
        index=class_names, 
        columns=class_names
    )
    np.fill_diagonal(df_cm.values, 0)
    top_10_errors = df_cm.stack().nlargest(10)
    
    print("--- Top 10 Most Confused Classes ---")
    print("(True Label) -> (Predicted Label) : Percent")
    for (true_label, pred_label), percentage in top_10_errors.items():
        print(f"({true_label}) -> ({pred_label}) : {percentage*100:.2f}%")
    return top_10_errors

def get_all_preds_and_probs(model, loader, device="cuda"):
    model.eval()

    all_logits = []
    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for x, y in tqdm(loader):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_logits.append(logits.cpu())
            all_probs.append(probs.cpu())
            all_preds.append(preds.cpu())
            all_labels.append(y.cpu())

    all_logits = torch.cat(all_logits)
    all_probs = torch.cat(all_probs)
    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    return all_labels, all_preds, all_probs, all_logits


def run_ser_experiment(model, dataset, predictor, layer_gc, device, num_samples=None):
    """Runs the main SER vs Uncertainty experiment."""
    results = {"set_size": [], "ser": [], "is_correct_pred": []}
    
    indices = range(len(dataset))
    if num_samples:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
        
    print(f"Running SER Experiment on {len(indices)} samples...")
    
    for idx in tqdm(indices):
        img, label, bbox = dataset[idx]
        input_tensor = img.unsqueeze(0).to(device)
        
        with torch.no_grad():
            prediction_set = predictor.predict(input_tensor)
            set_size = prediction_set.sum().item()
            
        logits = model(input_tensor)
        pred_class = logits.argmax(dim=1).item()
        is_correct = (pred_class == label)
        
        attr = layer_gc.attribute(input_tensor, target=pred_class, relu_attributions=True)
        attr_map = attr.squeeze().detach().cpu().numpy()
        if len(attr_map.shape) == 3: attr_map = np.mean(attr_map, axis=0)
            
        ser_score = calculate_ser_captum(attr_map, bbox, (448, 448))
        
        results["set_size"].append(set_size)
        results["ser"].append(ser_score)
        results["is_correct_pred"].append(is_correct)
        
    return pd.DataFrame(results)

def evaluate_class_coverage(predictor, dataloader, device):
    """Calculates coverage per class."""

    dataset = dataloader.dataset
    while hasattr(dataset, "dataset"): 
        dataset = dataset.dataset
        
    class_names = dataset.class_names

    n_classes = len(class_names)
    class_correct = {i: 0 for i in range(n_classes)}
    class_total = {i: 0 for i in range(n_classes)}
    
    print("Evaluating Class-Conditional Coverage...")
    with torch.no_grad():
        for x, y in tqdm(dataloader):
            x, y = x.to(device), y.to(device)
            sets = predictor.predict(x)
            for i in range(x.shape[0]):
                label = y[i].item()
                is_covered = sets[i, label].item()
                class_total[label] += 1
                class_correct[label] += is_covered
                
    class_coverage = []
    for i in range(n_classes):
        cov = class_correct[i] / class_total[i] if class_total[i] > 0 else 0.0
        class_coverage.append(cov)
    return np.array(class_coverage)

def evaluate_baselines(model, dataloader, layer_gc, predictor, device):
    """Runs Pointing Game and Naive Baseline."""
    print("Evaluating Baselines...")
    pointing_hits = 0
    total_samples = 0
    total_set_size = 0
    all_probs = []
    all_labels = []
    
    for img, label, bbox in tqdm(dataloader):
        img = img.to(device)
        logits = model(img)
        probs = torch.softmax(logits, dim=1)
        pred_class = logits.argmax(dim=1).item()
        
        all_probs.append(probs.detach().cpu())
        all_labels.append(label)
        
        # Pointing Game
        attr = layer_gc.attribute(img, target=pred_class, relu_attributions=True)
        attr = attr.squeeze().detach().cpu().numpy()
        if len(attr.shape) == 3: attr = np.mean(attr, axis=0)
        hit = calculate_pointing_game(attr, bbox, (448, 448))
        pointing_hits += hit
        
        # CP Efficiency
        with torch.no_grad():
            sets = predictor.predict(img)
            total_set_size += sets.sum().item()
        total_samples += 1
        
    pointing_acc = pointing_hits / total_samples
    avg_set_size = total_set_size / total_samples
    
    # Naive Baseline
    all_probs = torch.cat(all_probs)
    all_labels = torch.tensor(all_labels)
    target_cov = 0.95
    best_k = 200
    n = len(all_labels)
    
    for k in range(1, 201):
        _, top_k = all_probs.topk(k, dim=1)
        covered = torch.any(top_k == all_labels.unsqueeze(1), dim=1)
        if (covered.sum().item() / n) >= target_cov:
            best_k = k
            break
            
    return pointing_acc, best_k, avg_set_size