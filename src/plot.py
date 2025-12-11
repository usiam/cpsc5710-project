import torchvision.utils
import matplotlib.pyplot as plt
import numpy as np
from utils import unnormalize

def show_image_grid_with_labels(image_batch, labels, class_names_map, nrow=8):
    """Displays a grid of images with their class labels in the title."""
    
    # Un-normalize and create grid
    unnormalized_batch = unnormalize(image_batch)
    image_grid = torchvision.utils.make_grid(unnormalized_batch, nrow=nrow)
    
    # Transpose from [C, H, W] to [H, W, C] for matplotlib
    np_image_grid = image_grid.numpy().transpose((1, 2, 0))
    np_image_grid = np.clip(np_image_grid, 0, 1)

    # 3. Create the title string using the labels
    # We only show labels for the first row of images
    labels_to_show = labels[:nrow]
    label_names = []
    
    for l in labels_to_show:
        # Get 0-indexed label
        class_id = l.item()
        name = class_names_map[class_id]
        label_names.append(name)
        
    title = f"Labels: {', '.join(label_names)}"

    # 4. Plot
    plt.figure(figsize=(17, 6)) # Slightly wider for the title
    plt.imshow(np_image_grid)
    plt.title(title, fontsize=12)
    plt.axis('off')
    plt.show()

def plot_metrics_loss(history_df):
    # Create a 2x3 grid of plots
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Training and Evaluation Metrics Over Epochs')
    
    # 1. Plot Loss
    axes[0, 0].plot(history_df['train_loss'], label='Train Loss')
    axes[0, 0].plot(history_df['test_loss'], label='Test Loss')
    axes[0, 0].set_title('Loss')
    axes[0, 0].legend()
    
    # 2. Plot Accuracy
    axes[0, 1].plot(history_df['train_Accuracy'], label='Train Accuracy')
    axes[0, 1].plot(history_df['test_Accuracy'], label='Test Accuracy')
    axes[0, 1].set_title('Accuracy')
    axes[0, 1].legend()
    
    # 3. Plot F1-Score
    axes[0, 2].plot(history_df['train_F1Score'], label='Train F1 (Macro)')
    axes[0, 2].plot(history_df['test_F1Score'], label='Test F1 (Macro)')
    axes[0, 2].set_title('F1-Score (Macro)')
    axes[0, 2].legend()
    
    # 4. Plot Precision
    axes[1, 0].plot(history_df['train_Precision'], label='Train Precision (Macro)')
    axes[1, 0].plot(history_df['test_Precision'], label='Test Precision (Macro)')
    axes[1, 0].set_title('Precision (Macro)')
    axes[1, 0].legend()
    
    # 5. Plot Recall
    axes[1, 1].plot(history_df['train_Recall'], label='Train Recall (Macro)')
    axes[1, 1].plot(history_df['test_Recall'], label='Test Recall (Macro)')
    axes[1, 1].set_title('Recall (Macro)')
    axes[1, 1].legend()
    
    # 6. Plot AUROC
    axes[1, 2].plot(history_df['train_AUROC'], label='Train AUROC (Macro)')
    axes[1, 2].plot(history_df['test_AUROC'], label='Test AUROC (Macro)')
    axes[1, 2].set_title('AUROC (Macro)')
    axes[1, 2].legend()
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()