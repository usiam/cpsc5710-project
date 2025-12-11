import numpy as np
import cv2

def calculate_ser_captum(attribution_map, bbox, original_image_size):
    """Calculates Saliency Energy Ratio (SER)."""
    H, W = original_image_size
    
    heatmap_resized = cv2.resize(attribution_map, (W, H))
    heatmap_resized = np.maximum(heatmap_resized, 0) # ReLU
    
    total_energy = np.sum(heatmap_resized)
    if total_energy == 0:
        return 0.0
    heatmap_norm = heatmap_resized / total_energy
    
    x, y, w, h = map(int, bbox)
    mask = np.zeros((H, W))
    
    x_min, x_max = max(0, x), min(W, x + w)
    y_min, y_max = max(0, y), min(H, y + h)
    mask[y_min:y_max, x_min:x_max] = 1.0
    
    return np.sum(heatmap_norm * mask)

def calculate_pointing_game(heatmap, bbox, original_shape=(448, 448)):
    """Returns 1 if max point is inside bbox, else 0."""
    H, W = original_shape
    heatmap_resized = cv2.resize(heatmap, (W, H))
    
    _, _, _, max_loc = cv2.minMaxLoc(heatmap_resized)
    max_x, max_y = max_loc
    
    x, y, w, h = map(int, bbox)
    
    if (x <= max_x <= x + w) and (y <= max_y <= y + h):
        return 1
    return 0