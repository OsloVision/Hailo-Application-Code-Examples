#!/usr/bin/env python3
import argparse
import os
import sys
from loguru import logger
import queue
import threading
from types import SimpleNamespace
from post_process.postprocessing import inference_result_handler, decode_and_postprocess
from functools import partial
import time
from pathlib import Path
import numpy as np
import re
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from common.hailo_inference import HailoInfer

from common.toolbox import init_input_source, load_json_file, get_labels, visualize, preprocess, FrameRateTracker
from typing import Tuple, Optional

# OCR engine will be initialized lazily by the OCR worker
OCR_ENGINE = None  # type: Optional[object]

# Minimum confidence to consider an OCR result "high confidence" for saving the crop
OCR_SAVE_MIN_CONF = 0.7

def _sanitize_text_for_filename(text: str, max_len: int = 60) -> str:
    """Sanitize OCR text to be safe in filenames.

    - Keep alphanumerics, dash, underscore and space
    - Collapse whitespace to single underscore
    - Trim length
    - Fallback to 'text' if empty after sanitization
    """
    # Normalize spaces
    text = " ".join(text.strip().split())
    # Remove characters that are not safe
    safe = re.sub(r"[^A-Za-z0-9 _\-]", "", text)
    # Replace spaces with underscore
    safe = re.sub(r"\s+", "_", safe)
    # Trim
    if len(safe) > max_len:
        safe = safe[:max_len].rstrip("_-")
    # Fallback
    return safe or "text"


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

def inference_callback(
        completion_info,
        bindings_list: list,
        input_batch: list,
        output_queue: queue.Queue
) -> None:
    """
    infernce callback to handle inference results and push them to a queue.

    Args:
        completion_info: Hailo inference completion info.
        bindings_list (list): Output bindings for each inference.
        input_batch (list): Original input frames.
        output_queue (queue.Queue): Queue to push output results to.
    """
    if completion_info.exception:
        logger.error(f'Inference error: {completion_info.exception}')
    else:
        for i, bindings in enumerate(bindings_list):
            if len(bindings._output_names) == 1:
                result = bindings.output().get_buffer()
            else:
                result = {
                    name: np.expand_dims(
                        bindings.output(name).get_buffer(), axis=0
                    )
                    for name in bindings._output_names
                }
            output_queue.put((input_batch[i], result))


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
    Initialize queues, HailoAsyncInference instance, and run the inference.
    """
    config_data = load_json_file("config.json")
    labels = get_labels(labels_file)

    # Initialize input source from string: "camera", video file, or image folder
    cap, images = init_input_source(input_path, batch_size, resolution)


    input_queue = queue.Queue(maxsize=100)
    output_queue = queue.Queue(maxsize=100)
    ocr_queue = queue.Queue(maxsize=100)

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

    # Initialize OCR engine on startup to avoid first-use delays
    try:
        init_easyocr_engine()
    except Exception as e:
        logger.warning(f"EasyOCR init failed at startup: {e}")

    preprocess_thread = threading.Thread(
        target=preprocess,
        args=(images, cap, batch_size, input_queue, width, height)
    )


    postprocess_thread = threading.Thread(
        target=visualize_with_ocr,
        args=(output_queue, cap, save_stream_output, output_dir, post_process_callback_fn, ocr_queue)
    )

    ocr_thread = threading.Thread(
        target=ocr_worker,
        args=(ocr_queue, output_dir)
    )

    infer_thread = threading.Thread(
        target=infer, args=(hailo_inference, input_queue, output_queue)
    )


    preprocess_thread.start()
    postprocess_thread.start()
    infer_thread.start()
    ocr_thread.start()

    preprocess_thread.join()
    infer_thread.join()
    output_queue.put(None)  # Signal process thread to exit
    postprocess_thread.join()
    ocr_queue.put(None)  # Signal OCR thread to exit
    ocr_thread.join()

def visualize_with_ocr(output_queue, cap, save_stream_output, output_dir, callback, ocr_queue):
    """
    Like visualize, but crops detections and puts them in ocr_queue for OCR processing.
    """
    import cv2
    image_id = 0
    while True:
        result = output_queue.get()
        if result is None:
            break
        original, infer, *rest = result
        infer = infer[0] if isinstance(infer, list) and len(infer) == 1 else infer
        if rest:
            frame_with_detections = callback(original, infer, rest[0])
        else:
            frame_with_detections = callback(original, infer)

        # Process the raw inference results to get detection boxes
        config_data = load_json_file("config.json")
        decoded_detections = decode_and_postprocess(infer, config_data, "fast")  # Using "fast" arch
        
        # Check scores to see if threshold is reasonable
        if 'detection_scores' in decoded_detections:
            scores = decoded_detections['detection_scores']
            if len(scores) > 0:
                max_score = np.max(scores)
                score_threshold = config_data['fast']['score_threshold']
                logger.debug(f"Max detection score: {max_score:.3f}, threshold: {score_threshold}")
        
        # --- Crop detections and enqueue for OCR ---
        if isinstance(decoded_detections, dict) and 'detection_boxes' in decoded_detections:
            boxes = decoded_detections['detection_boxes']
        else:
            boxes = []
        
        
        if len(boxes) > 0:
            h, w = original.shape[:2]
            model_h, model_w = 640, 640  # Model input size
            
            # Calculate preprocessing scaling and padding (same as default_preprocess)
            scale = min(model_w / w, model_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            x_offset = (model_w - new_w) // 2
            y_offset = (model_h - new_h) // 2
            
            for i, box in enumerate(boxes):
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
                
                crop = original[ymin:ymax, xmin:xmax]
                
                if crop.size > 0:
                    try:
                        ocr_queue.put((crop, image_id), block=False)
                        logger.debug(f"Enqueued crop {i} for OCR processing")
                    except queue.Full:
                        print("WARNING: OCR queue full, dropping crop")

        image_id += 1
        # ...existing code for saving/displaying frame_with_detections...

def init_easyocr_engine() -> None:
    """Initialize the global EasyOCR engine at startup if available."""
    global OCR_ENGINE
    if OCR_ENGINE not in (None, -1):
        return
    try:
        logger.debug("Initializing EasyOCR engine at startup...")
        from easyocr import Reader  # type: ignore
        # Use CPU by default for widest compatibility; enable GPU if your env supports it
        OCR_ENGINE = Reader(['en'], gpu=False)
        logger.info("EasyOCR engine initialized (startup)")
    except Exception as e:
        OCR_ENGINE = -1
        logger.error(f"Failed to initialize EasyOCR at startup: {e}")

def process_ocr(crop_img) -> Tuple[str, float]:
    """
    Run OCR on a cropped image using EasyOCR.

    Args:
        crop_img (np.ndarray): BGR image of the cropped region.

    Returns:
        Tuple[str, float]: (recognized_text, confidence). Empty text with 0.0 if none.
    """

    logger.debug("Processing OCR...")
    import cv2
    global OCR_ENGINE

    # Lazy import to avoid hard dependency if OCR is not needed
    if OCR_ENGINE is None:
        logger.debug("Initializing EasyOCR engine (lazy)...")
        from easyocr import Reader  # type: ignore
        OCR_ENGINE = Reader(['en'], gpu=False)
        logger.debug("EasyOCR engine initialized (lazy)")


    if crop_img is None or crop_img.size == 0:
        return "", 0.0

    # Minimal pre-processing: ensure reasonable size for small text
    h, w = crop_img.shape[:2]
    if min(h, w) < 16:
        scale = max(1, int(16 / max(1, min(h, w))))
        crop_img = cv2.resize(crop_img, (w * scale, h * scale), interpolation=cv2.INTER_CUBIC)

    # Run OCR with a timeout to avoid hangs on first-time model downloads/init
    logger.debug(f"Running OCR on crop of size {w}x{h}")
    result_container = {}

    def _do_ocr():
        try:
            # EasyOCR returns list of [bbox, text, confidence]
            # detail=1 retains confidence; paragraph=False processes lines separately
            result_container['res'] = OCR_ENGINE.readtext(crop_img, detail=1, paragraph=False)
        except Exception as e:
            result_container['err'] = e

    t = threading.Thread(target=_do_ocr, daemon=True)
    t.start()
    t.join(1.5)  # seconds
    if t.is_alive():
        logger.warning("EasyOCR timed out on crop; skipping")
        return "", 0.0
    if 'err' in result_container:
        logger.debug(f"EasyOCR failed on crop: {result_container['err']}")
        return "", 0.0
    logger.debug(f"EasyOCR succeeded on crop: {result_container}")
    result = result_container.get('res', None)

    texts = []
    confs = []
    try:
        for item in (result or []):
            # item format: (bbox, text, confidence)
            if isinstance(item, (list, tuple)) and len(item) >= 3:
                text = item[1]
                conf = item[2]
                if isinstance(text, str):
                    texts.append(text)
                try:
                    confs.append(float(conf))
                except Exception:
                    pass
    except Exception as e:
        logger.debug(f"Unexpected EasyOCR result format: {e}; raw={result}")

    if not texts:
        return "", 0.0

    # Join multiple tokens; choose average confidence
    text = " ".join(t.strip() for t in texts if t)
    confidence = float(np.mean(confs)) if confs else 0.0
    return text, confidence

def ocr_worker(ocr_queue, output_dir: str):
    """
    Worker thread to run EasyOCR on crops from the ocr_queue.
    """
    import cv2
    logger.debug("OCR worker starting; EasyOCR will initialize on first use")
    # Prepare output sub-folder for OCR crops
    ocr_out_dir = os.path.join(output_dir or os.getcwd(), "ocr_crops")
    os.makedirs(ocr_out_dir, exist_ok=True)
    
    
    while True:
        item = ocr_queue.get()
        if item is None:
            logger.debug("OCR worker received stop signal")
            break
        crop, image_id = item
        logger.debug(f"Processing OCR for image {image_id}")
        
        # Apply OCR preprocessing and get result
        text, confidence = process_ocr(crop)

        # Log all OCR results at info level, tag low-confidence ones
        level_tag = "OK" if confidence > 0.3 else "LOW"
        logger.info(f"[OCR-{level_tag}] image_id={image_id} text={text!r} conf={confidence:.3f}")

        # Save high-confidence crops with text in filename
        try:
            if text and confidence >= OCR_SAVE_MIN_CONF:
                safe_text = _sanitize_text_for_filename(text)
                ts = int(time.time() * 1000)
                filename = f"ocr_{image_id}_{safe_text}_{int(confidence*100)}.png"
                save_path = os.path.join(ocr_out_dir, filename)
                cv2.imwrite(save_path, crop)
                logger.info(f"Saved OCR crop: {save_path}")
        except Exception as e:
            logger.warning(f"Failed to save OCR crop: {e}")
    
    logger.debug("OCR worker exiting")


def infer(hailo_inference, input_queue, output_queue):
    """
    Main inference loop that pulls data from the input queue, runs asynchronous
    inference, and pushes results to the output queue.

    Each item in the input queue is expected to be a tuple:
        (input_batch, preprocessed_batch)
        - input_batch: Original frames (used for visualization or tracking)
        - preprocessed_batch: Model-ready frames (e.g., resized, normalized)

    Args:
        hailo_inference (HailoInfer): The inference engine to run model predictions.
        input_queue (queue.Queue): Provides (input_batch, preprocessed_batch) tuples.
        output_queue (queue.Queue): Collects (input_frame, result) tuples for visualization.

    Returns:
        None
    """
    while True:
        next_batch = input_queue.get()
        if not next_batch:
            break  # Stop signal received

        input_batch, preprocessed_batch = next_batch
        # Prepare the callback for handling the inference result
        inference_callback_fn = partial(
            inference_callback,
            input_batch=input_batch,
            output_queue=output_queue
        )

        # Run async inference
        hailo_inference.run(preprocessed_batch, inference_callback_fn)

    # Release resources and context
    hailo_inference.close()



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
