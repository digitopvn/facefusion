# main.py
from fastapi import FastAPI, APIRouter, Body, File, UploadFile, Body
import os
import tempfile
import base64
from fastapi.middleware.cors import CORSMiddleware
from facefusion.api.security import SecurityMiddleware
from facefusion.core import conditional_process
import uvicorn
from typing import Optional
import numpy as np
import cv2
from dotenv import load_dotenv
import os
import json
import datetime
from facefusion import (
    state_manager,
    content_analyser,
    logger,
    face_classifier,
    face_detector,
    face_landmarker,
    face_masker,
    face_recognizer,
    voice_extractor,
)
from facefusion.vision import (
    detect_image_resolution,
    pack_resolution,
    read_static_image,
    write_image,
)
from facefusion.face_analyser import get_many_faces
from facefusion.vision import read_static_images
from facefusion.face_store import (
    clear_static_faces,
)
from facefusion.face_detector import detect_faces
from facefusion.face_classifier import classify_face
from facefusion.face_helper import apply_nms, get_nms_threshold
from facefusion.execution import get_available_execution_providers
from facefusion.processors.core import get_processors_modules

# Load environment variables
load_dotenv()

# Get environment variables with defaults
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("DEBUG_MODE", "False").lower() == "true"
CORS_ORIGINS = json.loads(os.getenv("CORS_ORIGINS", '["*"]'))
API_UPSCALE_URL = os.getenv("API_UPSCALE_URL", "http://localhost:3033")

ouputFolderDir = str(
    os.getenv("OUTPUT_FOLDER_DIR", os.path.join(os.getcwd(), "output"))
)


def initialize_facefusion():
    """Initialize facefusion with default state manager settings"""
    # Initialize execution providers
    available_execution_providers = get_available_execution_providers()
    default_execution_provider = (
        available_execution_providers[0] if available_execution_providers else "cpu"
    )

    # Set required state manager items with defaults
    state_manager.init_item(
        "download_scope", "full"
    )  # Use 'full' to ensure all models are downloaded
    state_manager.init_item("execution_providers", [default_execution_provider])
    state_manager.init_item("execution_device_ids", [0])
    state_manager.init_item("execution_thread_count", 8)
    state_manager.init_item("video_memory_strategy", "strict")
    state_manager.init_item("system_memory_limit", 0)
    state_manager.init_item("log_level", "info")

    # Initialize processors - default to face_swapper
    state_manager.init_item("processors", ["face_swapper"])

    # Face swapper specific settings
    state_manager.init_item("face_swapper_model", "hyperswap_1a_256")
    state_manager.init_item("face_swapper_pixel_boost", "256x256")
    state_manager.init_item("face_swapper_weight", 0.5)

    # Output settings
    state_manager.init_item("output_image_quality", 100)
    state_manager.init_item("output_image_scale", 1.0)
    state_manager.init_item("keep_temp", True)
    state_manager.init_item("temp_frame_format", "png")

    # Face detector settings
    state_manager.init_item("face_detector_model", "yolo_face")
    state_manager.init_item("face_detector_score", 0.5)
    state_manager.init_item("face_detector_angles", [0])

    # Face landmarker settings
    state_manager.init_item("face_landmarker_model", "2dfan4")

    # Face masker settings
    state_manager.init_item("face_occluder_model", "xseg_1")
    state_manager.init_item("face_parser_model", "bisenet_resnet_34")

    # Face selector settings
    state_manager.init_item("face_selector_mode", "reference")
    state_manager.init_item("face_selector_order", "best-worst")

    # Initialize logger
    logger.init(state_manager.get_item("log_level"))

    # Pre-check and download common modules
    common_modules = [
        ("content_analyser", content_analyser),
        ("face_classifier", face_classifier),
        ("face_detector", face_detector),
        ("face_landmarker", face_landmarker),
        ("face_masker", face_masker),
        ("face_recognizer", face_recognizer),
        ("voice_extractor", voice_extractor),
    ]

    logger.info("Initializing common modules...", __name__)
    for module_name, module in common_modules:
        logger.info(f"Initializing {module_name}...", __name__)
        if not module.pre_check():
            logger.error(f"{module_name} initialization failed", __name__)
            return False

    # Pre-check and download processor models
    logger.info("Initializing processors...", __name__)
    for processor_module in get_processors_modules(
        state_manager.get_item("processors")
    ):
        logger.info(f"Initializing {processor_module.__name__}...", __name__)
        if not processor_module.pre_check():
            logger.error(f"{processor_module.__name__} initialization failed", __name__)
            return False

    logger.info("FaceFusion API initialized successfully", __name__)

    for processor_module in get_processors_modules(
        state_manager.get_item("processors")
    ):
        if not processor_module.pre_process("output"):
            return 2
    return True


# Initialize on module load
_initialized = False
if not _initialized:
    _initialized = initialize_facefusion()
    if not _initialized:
        raise RuntimeError("Failed to initialize FaceFusion")


# Define the UTC+7 timezone
class UTCOffset(datetime.tzinfo):
    def __init__(self, offset_hours):
        self.offset = datetime.timedelta(hours=offset_hours)

    def utcoffset(self, dt):
        return self.offset

    def dst(self, dt):
        return datetime.timedelta(0)

    def tzname(self, dt):
        hours = int(self.offset.total_seconds() // 3600)
        return f"UTC+{hours}"


# Create a timezone object for UTC+7
tz_plus_7 = UTCOffset(7)


app = FastAPI(title="Your API Service")
router = APIRouter()

# Configure CORS with environment variables
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Security Middleware (hash validation)
# Exclude paths that don't require authentication
app.add_middleware(
    SecurityMiddleware, exclude_paths=["/", "/docs", "/openapi.json", "/redoc"]
)


def isFile(path):
    return os.path.splitext(path)[1] != ""


def to_base64_str(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode("utf-8")


def save_file(file_path: str, encoded_data: str):
    decoded_data = base64.b64decode(encoded_data)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, "wb") as file:
        file.write(decoded_data)


def make_tmp_dir():

    # Get the current time in UTC
    current_time_utc = datetime.datetime.now(datetime.timezone.utc)
    # Convert the current time to the desired timezone
    current_time_plus_7 = current_time_utc.astimezone(tz_plus_7)
    # Get the current date in the desired timezone
    current_date = current_time_plus_7.strftime("%y/%m/%d/%M")
    # Format the time as requested (hour 0-24) in the desired timezone
    formatted_time = current_time_plus_7.strftime("%M-%S-%f")[
        :-2
    ]  # Remove the last two digits of microseconds

    # Create the base directory path
    base_dir = os.path.join(ouputFolderDir, current_date)
    # Ensure the base directory exists
    os.makedirs(base_dir, exist_ok=True)

    # Create the temporary directory path
    tempDir = tempfile.mkdtemp(prefix=formatted_time + "-", dir=base_dir)
    return tempDir


@router.post("/check-face")
async def check_face(params=Body(...)) -> dict:
    """
    Check and extract faces from a base64-encoded image.

    Parameters:
    - source: Base64-encoded image string
    - source_extension: (optional) Image extension (jpg, png, etc.), default 'jpg'
    - return_faces: (optional) If True, returns base64-encoded face images, default True
    - save_faces: (optional) If True, saves faces to temp directory and returns paths, default True
    - padding: (optional) Padding in pixels around face bounding box, default 30
    - include_attributes: (optional) If True, includes gender/age/race (adds ~30-50ms per face), default True
    - min_score: (optional) Minimum confidence score for face detection (0.0-1.0), default 0.5

    Returns:
    - status: 1 for success, 0 for error
    - data: Object containing:
      - count: Number of faces detected
      - faces: Array of face data with images/paths (original ratio maintained)
    """
    temp_dir = None
    try:
        # Get parameters
        source = params["source"]
        source_extension = params.get("source_extension", "jpg").lstrip(".")
        return_faces = params.get("return_faces", True)
        save_faces = params.get("save_faces", True)
        padding = params.get("padding", 30)
        include_attributes = params.get("include_attributes", True)
        min_score = params.get(
            "min_score", 0.5
        )  # Confidence threshold for face detection

        # Decode base64 directly to image (faster than file I/O)
        decoded_data = base64.b64decode(source)
        nparr = np.frombuffer(decoded_data, np.uint8)
        vision_frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if vision_frame is None:
            return {"status": 0, "message": "INVALID_IMAGE"}

        # Only create temp_dir if we need to save faces
        if save_faces:
            temp_dir = make_tmp_dir()

        # Lightweight face detection (skips embedding and classification for speed)
        all_bounding_boxes, all_face_scores, all_face_landmarks_5 = detect_faces(
            vision_frame
        )

        # Apply NMS (Non-Maximum Suppression) to filter overlapping detections and low-confidence faces
        nms_threshold = get_nms_threshold(
            state_manager.get_item("face_detector_model"),
            state_manager.get_item("face_detector_angles") or [0],
        )

        # Filter by confidence score and remove overlapping detections
        keep_indices = apply_nms(
            all_bounding_boxes, all_face_scores, min_score, nms_threshold
        )

        # Keep only the high-confidence, non-overlapping faces
        bounding_boxes = [all_bounding_boxes[i] for i in keep_indices]
        face_scores = [all_face_scores[i] for i in keep_indices]
        face_landmarks_5 = [all_face_landmarks_5[i] for i in keep_indices]

        face_count = len(bounding_boxes)
        logger.info(
            f"Detected {face_count} faces (filtered from {len(all_bounding_boxes)} raw detections)",
            __name__,
        )

        if face_count == 0:
            return {"status": 0, "message": "FACE_NO_FOUND"}

        # Basic response
        response_data = {
            "count": face_count,
            "has_single_face": face_count == 1,
            "has_multiple_faces": face_count > 1,
        }

        # Extract and return faces if requested
        if return_faces or save_faces:
            face_data = []
            img_height, img_width = vision_frame.shape[:2]

            for idx, bounding_box in enumerate(bounding_boxes):
                # Get bounding box coordinates
                x1, y1, x2, y2 = bounding_box

                # Add padding and ensure within image boundaries
                x1_padded = max(0, int(x1 - padding))
                y1_padded = max(0, int(y1 - padding))
                x2_padded = min(img_width, int(x2 + padding))
                y2_padded = min(img_height, int(y2 + padding))

                # Extract face region with padding (maintains original ratio)
                crop_vision_frame = vision_frame[
                    y1_padded:y2_padded, x1_padded:x2_padded
                ].copy()

                face_info = {
                    # "index": idx,
                    # "bounding_box": bounding_box.tolist(),
                    # "bounding_box_with_padding": [x1_padded, y1_padded, x2_padded, y2_padded],
                    "width": x2_padded - x1_padded,
                    "height": y2_padded - y1_padded,
                    "score": float(face_scores[idx]),  # Detection confidence score
                }

                # Add gender/age/race if requested (adds ~30-50ms per face)
                if include_attributes:
                    gender, age, race = classify_face(
                        vision_frame, face_landmarks_5[idx]
                    )
                    face_info["gender"] = gender
                    face_info["age"] = list(age) if age else None
                    face_info["race"] = race

                    # Return as base64
                    # if return_faces:
                    # _, buffer = cv2.imencode('.jpg', crop_vision_frame)
                    # face_base64 = base64.b64encode(buffer).decode('utf-8')
                    # face_info["image"] = face_base64

                    # Save to file
                    if save_faces:
                        face_filename = f"face_{idx}_{gender}_{age[0]}.jpg"
                        face_path = os.path.join(str(temp_dir), str(face_filename))
                        write_image(face_path, crop_vision_frame)
                        face_info["path"] = face_path.replace("\\", "/").replace(
                            ouputFolderDir, ""
                        )

                face_data.append(face_info)

            response_data["faces"] = face_data
        print(response_data)
        return {"status": 1, "data": response_data}

    except ValueError as e:
        logger.error(f"ValueError in check_face: {str(e)}", __name__)
        return {"status": 0, "message": "FACE_NO_FOUND", "error": str(e)}
    except Exception as e:
        logger.error(f"Exception in check_face: {str(e)}", __name__)
        return {"status": 0, "message": "FACE_NO_FOUND", "error": str(e)}


@app.post("/")
async def process_frames(params=Body(...)) -> dict:
    output_path = None  # Initialize to None to handle exceptions properly

    try:
        sources = params["sources"]
        source_extension = params.get("source_extension", "jpg").lstrip(
            "."
        )  # Remove leading dot if present
        target_extension = params.get("target_extension", "jpg").lstrip(
            "."
        )  # Remove leading dot if present
        source_paths = []

        temp_dir = make_tmp_dir()

        print(f"Temp Dir: {temp_dir}")

        # current_time = datetime.datetime.now()
        # formatted_time = current_time.strftime("%Y%m%d%H%M%S%f")[:-2]  # Remove the last two digits of microseconds

        # Get the current time in UTC
        current_time_utc = datetime.datetime.now(datetime.timezone.utc)
        # Convert the current time to the desired timezone
        current_time_plus_7 = current_time_utc.astimezone(tz_plus_7)
        # Get the current date in the desired timezone
        # current_date = current_time_plus_7.strftime("%Y-%m-%d")
        # Format the time as requested (hour 0-24) in the desired timezone
        formatted_time = current_time_plus_7.strftime("%H-%M-%S-%f")[
            :-2
        ]  # Remove the last two digits of microseconds

        for i, source in enumerate(sources):
            source_path = os.path.join(
                temp_dir,
                f"source{i}-{formatted_time}.{source_extension}",
            )
            source_paths.append(source_path)
            save_file(source_path, source)

        target = params["target"]
        target_path = os.path.join(
            temp_dir, f"target-{formatted_time}.{target_extension}"
        )

        save_file(target_path, target)

        output_path = os.path.join(temp_dir, f"output.{target_extension}")

        state_manager.set_item("source_paths", source_paths)
        state_manager.set_item("target_path", target_path)
        state_manager.set_item("output_path", output_path)

        output_image_resolution = detect_image_resolution(target_path)
        # output_image_resolutions = create_image_resolutions(output_image_resolution)
        if output_image_resolution is not None:
            state_manager.set_item(
                "output_image_resolution", pack_resolution(output_image_resolution)
            )
        else:
            logger.error("Failed to detect image resolution for target_path", __name__)
            return {"status": 0, "message": "Failed to detect image resolution."}

        conditional_process()
        clear_static_faces()
        # clear_reference_faces()

        return {
            "status": 1,
            "data": {
                # "output": output_upscale_base64,
                "local_path": output_path.replace(ouputFolderDir, "")
            },
        }

    except Exception as e:
        logger.error(f"Processing error: {str(e)}", __name__)
        clear_static_faces()
        # clear_reference_faces()
        # Return error response if output_path is not set or file doesn't exist
        if output_path and os.path.exists(output_path):
            return {"status": 1, "data": {"output": to_base64_str(output_path)}}
        else:
            return {"status": 0, "message": f"Processing failed: {str(e)}"}


@app.get("/")
async def home():
    return 1


def launch():
    app.include_router(router)
    uvicorn.run(app, host=HOST, port=PORT)
