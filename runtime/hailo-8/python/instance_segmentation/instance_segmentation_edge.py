#!/usr/bin/env python3
import argparse
import os
import sys
from loguru import logger
from types import SimpleNamespace
from post_process.postprocessing import inference_result_handler, decode_and_postprocess
from functools import partial
import time
from pathlib import Path
import numpy as np
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.hailo_inference import HailoInfer

from common.toolbox import init_input_source, load_json_file, get_labels, default_preprocess
from typing import Tuple, Optional


def parse_args() -> argparse.Namespace:
    """
    Initialize argument parser for the script.
    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Instance segmentation with detection crop saving - supporting Yolov5, Yolov8, and FastSAM architectures."
    )

    parser.add_argument(
        "-n", "--net",
        help="Path for the network in HEF format.",
        required=True
    )
    parser.add_argument(
        "-i", "--input",
        default="zidane.jpg",
        help="Path to the input - either an image or a folder of images."
    )
    parser.add_argument(
        "-b", "--batch_size",
        type=int,
        default=1,
        help="Number of images in one batch"
    )
    parser.add_argument(
        "-a", "--arch",
        required=True,
        help="The architecture type of the model: v5, v8 or fast"
    )
    parser.add_argument(
        "-l", "--labels",
        default=str(Path(__file__).parent.parent / "common" / "coco.txt"),
        help="Path to a text file containing labels. If not provided, coco2017 will be used."
    )
    parser.add_argument(
        "-o", "--output-dir",
        default=None,
        help="Directory to save the results."
    )
    parser.add_argument(
        "-r", "--resolution",
        choices=["sd", "hd", "fhd"],
        default=None,
        help="(Camera input only) Choose output resolution: 'sd' (640x480), 'hd' (1280x720), or 'fhd' (1920x1080). "
             "If not specified, the camera's native resolution will be used."
    )

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.net):
        raise FileNotFoundError(f"Network file not found: {args.net}")

    if args.output_dir is None:
        args.output_dir = os.path.join(os.getcwd(), "output")
    os.makedirs(args.output_dir, exist_ok=True)

    logger.debug("Starting up")

    return args


def save_detection_crops(original_frame, inference_result, config_data, arch, output_dir, frame_id):
    """
    Extract detection bounding boxes from inference results and save cropped regions as image files.
    
    Args:
        original_frame: Original input frame/image
        inference_result: Raw inference output from the model
        config_data: Configuration data loaded from config.json
        arch: Model architecture ('v5', 'v8', or 'fast')
        output_dir: Directory to save crop images
        frame_id: Unique identifier for the frame/image
    """
    import cv2
    
    # Process the raw inference results to get detection boxes
    decoded_detections = decode_and_postprocess(inference_result, config_data, arch)
    
    # Check if we have valid detections
    if not isinstance(decoded_detections, dict) or 'detection_boxes' not in decoded_detections:
        logger.debug(f"No detection boxes found in frame {frame_id}")
        return 0
    
    boxes = decoded_detections['detection_boxes']
    scores = decoded_detections.get('detection_scores', [])
    class_ids = decoded_detections.get('detection_classes', [])
    
    if len(boxes) == 0:
        logger.debug(f"No detections found in frame {frame_id}")
        return 0
    
    # Create crops directory
    crops_dir = os.path.join(output_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    
    h, w = original_frame.shape[:2]
    model_h, model_w = 640, 640  # Model input size
    
    # Calculate preprocessing scaling and padding (same as default_preprocess)
    scale = min(model_w / w, model_h / h)
    new_w, new_h = int(w * scale), int(h * scale)
    x_offset = (model_w - new_w) // 2
    y_offset = (model_h - new_h) // 2
    
    crop_count = 0
    
    for i, box in enumerate(boxes):
        # Get confidence score and class if available
        confidence = scores[i] if i < len(scores) else 1.0
        class_id = int(class_ids[i]) if i < len(class_ids) else 0
        
        # Boxes are normalized to model input size (640x640)
        # Format: [xmin, ymin, xmax, ymax] normalized 0-1
        xmin_norm, ymin_norm, xmax_norm, ymax_norm = box
        
        # Scale to model coordinates (640x640)
        xmin_model = xmin_norm * model_w
        ymin_model = ymin_norm * model_h
        xmax_model = xmax_norm * model_w
        ymax_model = ymax_norm * model_h
        
        # Remove padding to get scaled image coordinates
        xmin_scaled = xmin_model - x_offset
        ymin_scaled = ymin_model - y_offset
        xmax_scaled = xmax_model - x_offset
        ymax_scaled = ymax_model - y_offset
        
        # Scale back to original image coordinates
        xmin = int(xmin_scaled / scale)
        ymin = int(ymin_scaled / scale)
        xmax = int(xmax_scaled / scale)
        ymax = int(ymax_scaled / scale)
        
        # Clamp coordinates to image bounds
        xmin = max(0, min(xmin, w-1))
        xmax = max(0, min(xmax, w-1))
        ymin = max(0, min(ymin, h-1))
        ymax = max(0, min(ymax, h-1))
        
        # Skip if crop is too small
        if (xmax - xmin) < 10 or (ymax - ymin) < 10:
            continue
            
        # Extract crop
        crop = original_frame[ymin:ymax, xmin:xmax]
        
        if crop.size > 0:
            # Create filename with frame_id, crop_id, class_id, and confidence
            crop_filename = f"frame_{frame_id:06d}_crop_{i:03d}_class_{class_id}_conf_{confidence:.3f}.jpg"
            crop_path = os.path.join(crops_dir, crop_filename)
            
            # Save crop
            cv2.imwrite(crop_path, crop)
            crop_count += 1
            logger.debug(f"Saved crop: {crop_filename}")
    
    return crop_count


def run_inference_pipeline(
    net,
    input_path,
    arch,
    batch_size,
    labels_file,
    output_dir,
    save_stream_output=False,
    resolution="sd",
    enable_tracking=False,
    show_fps=False
) -> None:
    """
    Run synchronous inference pipeline - read frame, infer, process, save crops.
    No video output is generated, only detection crops are saved.
    """
    import cv2
    from common.toolbox import default_preprocess
    
    config_data = load_json_file("config.json")
    labels = get_labels(labels_file)

    # Initialize input source from string: "camera", video file, or image folder
    cap, images = init_input_source(input_path, batch_size, resolution)

    hailo_inference = HailoInfer(
        net,
        batch_size,
        output_type="FLOAT32")

    height, width, _ = hailo_inference.get_input_shape()
    
    frame_count = 0
    total_crops = 0
    
    try:
        if cap is not None:
            # Process video/camera frames
            logger.info("Processing video/camera frames and saving detection crops...")
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preprocessed_frame = default_preprocess(rgb_frame, width, height)
                
                # Run inference synchronously
                result = hailo_inference.run_sync([preprocessed_frame])
                
                # Save detection crops
                crop_count = save_detection_crops(
                    frame, result[0], config_data, arch, output_dir, frame_count
                )
                total_crops += crop_count
                
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames, saved {total_crops} crops so far")
                    
        else:
            # Process image files
            logger.info("Processing image files and saving detection crops...")
            for i, image in enumerate(images):
                # Preprocess image
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                preprocessed_image = default_preprocess(rgb_image, width, height)
                
                # Run inference synchronously
                result = hailo_inference.run_sync([preprocessed_image])
                
                # Save detection crops
                crop_count = save_detection_crops(
                    image, result[0], config_data, arch, output_dir, i
                )
                total_crops += crop_count
                
                logger.info(f"Processed image {i}, saved {crop_count} crops")
                
    finally:
        # Cleanup
        if cap is not None:
            cap.release()
        hailo_inference.close()
        
        logger.info(f"Processing completed. Total frames: {frame_count}, Total crops saved: {total_crops}")
        logger.info(f"Crops saved to: {os.path.join(output_dir, 'crops')}")


def main() -> None:
    args = parse_args()
    run_inference_pipeline(
        args.net,
        args.input,
        args.arch,
        args.batch_size,
        args.labels,
        args.output_dir,
        False,  # save_stream_output is always False for this version
        args.resolution
    )


if __name__ == "__main__":
    main()