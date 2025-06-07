import os
import numpy as np
import torch
import cv2
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import json

sam2_checkpoint = "/opt/sam2/checkpoints/sam2.1_hiera_small.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
device = torch.device("cpu")

# Initialize model
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=device)
predictor = SAM2ImagePredictor(sam2_model)

def show_masks(image, masks, scores):
    results = []
    for i, (mask, score) in enumerate(zip(masks, scores)):
        contour_data = show_mask(mask, i)
        results.append({
            "mask_id": i,
            "score": float(score),
            "contours": contour_data
        })
    return results


def contour_to_dbgimage(mask, contours, i):
    np.random.seed(3)
    
    h, w = mask.shape[-2:]

    # Create an empty transparent image for drawing contours
    mask_image = np.zeros((h, w, 4), dtype=np.float32)

    color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)

    for idx, contour in enumerate(contours):
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
        mask_image = cv2.drawContours(mask_image, contours, idx, color, thickness=2) 

    # Convert numpy array to PIL Image and save
    mask_pil = Image.fromarray((mask_image * 255).astype(np.uint8))
    mask_pil.save(f"dbg/mask_{i}.png")


def show_mask(mask, i):
    mask = mask.astype(np.uint8)

    # Extract contours from mask.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    # Filter out small contours
    if contours:
        # Calculate contour areas
        areas = [cv2.contourArea(contour) for contour in contours]
        # Get median area as reference
        median_area = np.median(areas)
        # Filter contours that are significantly smaller than the median
        contours = [contour for contour, area in zip(contours, areas) if area > median_area * 0.1]

    # Try to smooth contours
    contours = [cv2.approxPolyDP(contour, epsilon=0.01, closed=True) for contour in contours]

    contour_data = []
    for idx, contour in enumerate(contours):
        # Flatten contour points
        points_array = contour.reshape(-1, 2).flatten().tolist()
        contour_data.append({
            "contour_id": idx,
            "points": points_array
        })
        
        # # Save the points array to a file for later use
        # with open(f"contour_{i}_{idx}.json", "w") as f:
        #     json.dump(points_array, f)

    contour_to_dbgimage(mask, contours, i)
    return contour_data


def detect_segments(pil_image, point, props=None):
    """
    Detect segments in an image using a pre-trained model.
    """
    # Default point if not provided in props
    input_point = np.array([point])
    input_label = np.array([1])
    image = np.array(pil_image.convert("RGB"))

    predictor.set_image(image)

    # If props contains point information, use it
    if props and 'points' in props and len(props['points']) > 0:
        points = props['points']
        input_point = np.array(points)
        input_label = np.array([1] * len(points))

    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    sorted_ind = np.argsort(scores)[::-1]
    masks = masks[sorted_ind]
    scores = scores[sorted_ind]
    logits = logits[sorted_ind]

    return show_masks(image, masks, scores)
