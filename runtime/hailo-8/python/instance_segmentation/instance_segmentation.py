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
        description="Instance segmentation supporting Yolov5, Yolov8, and FastSAM architectures."
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
        "-s", "--save_stream_output",
        action="store_true",
        help="Save the output of the inference from a stream."
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
    Run synchronous inference pipeline - read frame, infer, process, save.
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

    post_process_callback_fn = partial(
        inference_result_handler,
        config_data=config_data,
        arch=arch,
        labels=labels,
        nms_postprocess_enabled=hailo_inference.is_nms_postprocess_enabled()
    )

    height, width, _ = hailo_inference.get_input_shape()
    
    # Setup video writer for MP4 output if processing video
    out = None
    if cap is not None and save_stream_output:
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, "output.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(out_path, fourcc, fps, (frame_width, frame_height))
        logger.info(f"Saving video output to: {out_path}")

    frame_count = 0
    
    try:
        if cap is not None:
            # Process video/camera frames
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Preprocess frame
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                preprocessed_frame = default_preprocess(rgb_frame, width, height)
                
                # Run inference synchronously
                result = hailo_inference.run_sync([preprocessed_frame])
                
                # Process results and create annotated frame
                annotated_frame = post_process_callback_fn(frame, result[0])
                
                # Save or display output
                if save_stream_output and out is not None:
                    out.write(annotated_frame)
                else:
                    # Save individual frame
                    cv2.imwrite(os.path.join(output_dir, f"frame_{frame_count:06d}.jpg"), annotated_frame)
                
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")
                    
        else:
            # Process image files
            for i, image in enumerate(images):
                # Preprocess image
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                preprocessed_image = default_preprocess(rgb_image, width, height)
                
                # Run inference synchronously
                result = hailo_inference.run_sync([preprocessed_image])
                
                # Process results and create annotated image
                annotated_image = post_process_callback_fn(image, result[0])
                
                # Save output
                output_path = os.path.join(output_dir, f"output_{i}.png")
                cv2.imwrite(output_path, annotated_image)
                logger.info(f"Saved annotated image: {output_path}")
                
    finally:
        # Cleanup
        if out is not None:
            out.release()
        if cap is not None:
            cap.release()
        hailo_inference.close()
        
        logger.info(f"Processing completed. Total frames: {frame_count}")


def main() -> None:
    args = parse_args()
    run_inference_pipeline(
        args.net,
        args.input,
        args.arch,
        args.batch_size,
        args.labels,
        args.output_dir,
        args.save_stream_output,
        args.resolution
    )


if __name__ == "__main__":
    main()
