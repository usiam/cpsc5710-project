import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import pandas as pd
import numpy as np
import cv2
import torch
from torchvision import transforms
from captum.attr import visualization as viz
from xai_utils import calculate_ser_captum # Import metric
import math

# Ensure plots directory exists
os.makedirs("../results/plots", exist_ok=True)

def plot_efficiency_curve(df, predictor_type="SplitPredictor"):
    plt.figure(figsize=(7, 5))
    subset = df[df["Predictor"] == predictor_type]
    sns.lineplot(data=subset, x="Target_Coverage", y="average_size", hue="Score_Function", style="Score_Function", markers=True)
    plt.title(f"Efficiency Curve: {predictor_type}\n(Lower is Better)", fontsize=14)
    plt.xlabel("Target Coverage ($1-\\alpha$)")
    plt.ylabel("Average Set Size")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.savefig(f"../results/plots/efficiency_curve_{predictor_type}.png", dpi=300)
    plt.show()

def plot_calibration_curve(df, predictor_type="SplitPredictor"):
    plt.figure(figsize=(7, 5))
    subset = df[df["Predictor"] == predictor_type]
    sns.lineplot(data=subset, x="Target_Coverage", y="coverage_rate", hue="Score_Function", style="Score_Function", markers=True)
    plt.plot([df["Target_Coverage"].min(), df["Target_Coverage"].max()], [df["Target_Coverage"].min(), df["Target_Coverage"].max()], 'k--', label="Ideal")
    plt.title(f"Calibration Plot: {predictor_type}", fontsize=14)
    plt.tight_layout()
    plt.savefig(f"../results/plots/calibration_curve_{predictor_type}.png", dpi=300)
    plt.show()

def visualize_and_save_case(index, category_name, dataset, model, predictor, layer_gc, device):
    """Single case study visualization (GradCAM + BBox)."""
    img, label, bbox = dataset[index]
    input_tensor = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction_set = predictor.predict(input_tensor)
        set_size = prediction_set.sum().item()
        
    logits = model(input_tensor)
    pred_class = logits.argmax(dim=1).item()
    attr = layer_gc.attribute(input_tensor, target=pred_class, relu_attributions=True)
    attr = attr.squeeze().detach().cpu().numpy()
    if len(attr.shape) == 3: attr = np.mean(attr, axis=0)
    
    ser = calculate_ser_captum(attr, bbox, (448, 448))
    
    inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    img_disp = inv_normalize(img).permute(1, 2, 0).numpy()
    img_disp = np.clip(img_disp, 0, 1)

    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(img_disp)
    heatmap_resized = cv2.resize(attr, (448, 448))
    viz.visualize_image_attr(np.expand_dims(heatmap_resized, 2), img_disp, method="blended_heat_map", sign="positive", show_colorbar=False, plt_fig_axis=(fig, ax), use_pyplot=False)
    
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=3, edgecolor='lime', facecolor='none', label='GT BBox')
    ax.add_patch(rect)
    status = "Correct" if prediction_set[0, label].item() else "Missed"
    ax.set_title(f"{category_name}\nSet Size: {set_size} | SER: {ser:.2f} | {status}", fontsize=14)
    ax.axis('off')
    plt.savefig(f"../results/plots/case_{category_name.lower().replace(' ', '_')}.png", bbox_inches='tight')
    plt.close()
    print(f"Saved case study: {category_name}")

def visualize_multi_level(index, category_name, dataset, model, predictor, layer_gc, gb, device):
    """Multi-panel visualization (Original, GradCAM, GuidedBackprop)."""
    img, label, bbox = dataset[index]
    input_tensor = img.unsqueeze(0).to(device)
    
    with torch.no_grad():
        prediction_set = predictor.predict(input_tensor)
        set_size = prediction_set.sum().item()
        
    logits = model(input_tensor)
    pred_class = logits.argmax(dim=1).item()
    
    # GradCAM
    attr_gc = layer_gc.attribute(input_tensor, target=pred_class, relu_attributions=True)
    attr_gc = attr_gc.squeeze().detach().cpu().numpy()
    if len(attr_gc.shape) == 3: attr_gc = np.mean(attr_gc, axis=0)
    heatmap_gc = cv2.resize(attr_gc, (448, 448))
    
    # Guided Backprop
    attr_gb = gb.attribute(input_tensor, target=pred_class)
    attr_gb = attr_gb.squeeze().cpu().detach().numpy()
    attr_gb = np.transpose(attr_gb, (1, 2, 0))
    
    # Plotting
    inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], std=[1/0.229, 1/0.224, 1/0.225])
    img_disp = inv_normalize(img).permute(1, 2, 0).numpy()
    img_disp = np.clip(img_disp, 0, 1)

    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    
    ax[0].imshow(img_disp)
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='lime', facecolor='none')
    ax[0].add_patch(rect)
    ax[0].set_title(f"Input: {category_name}\nSet Size: {set_size}", fontsize=14)
    ax[0].axis('off')
    
    viz.visualize_image_attr(np.expand_dims(heatmap_gc, 2), img_disp, method="blended_heat_map", sign="positive", show_colorbar=True, plt_fig_axis=(fig, ax[1]), use_pyplot=False)
    ax[1].set_title("Level 1: Region Focus (GradCAM)", fontsize=14)

    viz.visualize_image_attr(attr_gb, img_disp, method="heat_map", sign="absolute_value", show_colorbar=True, outlier_perc=2, plt_fig_axis=(fig, ax[2]), use_pyplot=False)
    ax[2].set_title("Level 2: Fine Features (GuidedBackprop)", fontsize=14)
    
    plt.savefig(f"../results/plots/multilevel_gb_{category_name.lower().replace(' ', '_')}.png", bbox_inches='tight')
    plt.close()
    print(f"Saved multilevel plot: {category_name}")

def visualize_prediction_set(index, dataset, model, predictor, layer_gc, device, class_names):
    """
    Visualizes the GradCAM explanation for EVERY class in the Conformal Prediction Set.
    Helps diagnose why the model is uncertain between specific options.
    """
    # 1. Load Data
    img, label, bbox = dataset[index]
    input_tensor = img.unsqueeze(0).to(device)
    
    # 2. Get Conformal Prediction Set
    with torch.no_grad():
        prediction_set = predictor.predict(input_tensor) # Binary vector
        candidate_indices = torch.nonzero(prediction_set.flatten()).flatten().cpu().numpy()
        
    set_size = len(candidate_indices)
    print(f"Prediction Set Size: {set_size}")
    print(f"Candidates: {[class_names[i] for i in candidate_indices]}")
    
    # 3. Setup Plotting Grid
    # We want a grid roughly square (e.g., 4 items -> 2x2)
    cols = min(set_size + 1, 4) # +1 for original image
    rows = math.ceil((set_size + 1) / cols)
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if isinstance(axes, np.ndarray): axes = axes.flatten()
    else: axes = [axes]
    
    # 4. Plot Original Image
    inv_normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    img_disp = inv_normalize(img).permute(1, 2, 0).numpy()
    img_disp = np.clip(img_disp, 0, 1)
    
    axes[0].imshow(img_disp)
    rect = patches.Rectangle((bbox[0], bbox[1]), bbox[2], bbox[3], linewidth=2, edgecolor='lime', facecolor='none')
    axes[0].add_patch(rect)
    axes[0].set_title(f"True: {class_names[label]}\n(Set Size: {set_size})", fontweight='bold')
    axes[0].axis('off')
    
    # 5. Loop through Candidates and Explain Each
    for i, class_idx in enumerate(candidate_indices):
        ax_idx = i + 1
        if ax_idx >= len(axes): break
        
        # Generate GradCAM for THIS specific candidate class
        # (This asks: "What evidence do you see for Class X?")
        attr = layer_gc.attribute(input_tensor, target=int(class_idx), relu_attributions=True)
        attr = attr.squeeze().detach().cpu().numpy()
        if len(attr.shape) == 3: attr = np.mean(attr, axis=0)
    
        heatmap_resized = cv2.resize(attr, (448, 448))
        
        viz.visualize_image_attr(
            np.expand_dims(heatmap_resized, 2), 
            img_disp, 
            method="blended_heat_map", 
            sign="positive", 
            show_colorbar=False, 
            plt_fig_axis=(fig, axes[ax_idx]),
            use_pyplot=False
        )
        
        is_gt = (class_idx == label)
        color = 'green' if is_gt else 'red'
        name = class_names[class_idx]
        
        axes[ax_idx].set_title(f"{name}\n({'Correct' if is_gt else 'Alternative'})", color=color, fontweight='bold')
        axes[ax_idx].axis('off')

    for j in range(set_size + 1, len(axes)):
        axes[j].axis('off')
        
    plt.tight_layout()
    plt.savefig(f"../results/plots/prediction_set_viz_{index}.png")
    plt.show()

