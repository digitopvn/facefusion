# main.py
from fastapi import FastAPI, APIRouter, Body, File, UploadFile, Body
import os
import tempfile
import base64
from fastapi.middleware.cors import CORSMiddleware
from facefusion.core import conditional_process
import uvicorn
from typing import Optional
import numpy as np
import cv2
from dotenv import load_dotenv
import os
import json
import datetime
from facefusion import state_manager
from facefusion.vision import (
    create_image_resolutions,
    detect_image_resolution,
    pack_resolution,
)
from facefusion.face_analyser import get_many_faces
from facefusion.vision import read_static_images
import requests

# Load environment variables
load_dotenv()

# Get environment variables with defaults
HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("API_PORT", "8000"))
DEBUG = os.getenv("DEBUG_MODE", "False").lower() == "true"
CORS_ORIGINS = json.loads(os.getenv("CORS_ORIGINS", '["*"]'))
ouputFolderDir = str(
    os.getenv("OUTPUT_FOLDER_DIR", os.path.join(os.getcwd(), "output"))
)

# state_manager.set_item("processors", ["face_swapper", "face_enhancer"])
# state_manager.set_item("face_enhancer_model", "gfpgan_1.4")
# state_manager.set_item("face_swapper_model", "simswap_256")
# state_manager.set_item("face_enhancer_blend", 100)


# Define the UTC+7 timezone
class UTCOffset(datetime.tzinfo):
    def __init__(self, offset_hours):
        self.offset = datetime.timedelta(hours=offset_hours)

    def utcoffset(self, dt):
        return self.offset

    def dst(self, dt):
        return datetime.timedelta(0)

    def tzname(self, dt):
        return f"UTC+{self.offset.hours}"


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
    current_date = current_time_plus_7.strftime("%Y-%m-%d")
    # Format the time as requested (hour 0-24) in the desired timezone
    formatted_time = current_time_plus_7.strftime("%H-%M-%S-%f")[
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
    try:
        face_location = get_many_faces(read_static_images([params["source_path"]]))
        if len(face_location) == 1:
            return {"status": 1, "data": True}
        if len(face_location) > 1:
            return {"status": 0, "message": "MULTI_FACE"}

        return {"status": 0, "message": "FACE_NO_FOUND"}
    except ValueError as e:
        print(e)
        return {"status": 0, "message": "FACE_NO_FOUND"}
    except Exception as e:
        print(e)
        return {"status": 0, "message": "FACE_NO_FOUND"}


@app.post("/")
async def process_frames(params=Body(...)) -> dict:
    sources = params["sources"]
    source_extension = params["source_extension"]
    target_extension = params["target_extension"]
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
    current_date = current_time_plus_7.strftime("%Y-%m-%d")
    # Format the time as requested (hour 0-24) in the desired timezone
    formatted_time = current_time_plus_7.strftime("%H-%M-%S-%f")[
        :-2
    ]  # Remove the last two digits of microseconds

    for i, source in enumerate(sources):
        source_path = os.path.join(
            temp_dir,
            os.path.basename(f"source{i}-{formatted_time}.{source_extension}"),
        )
        source_paths.append(source_path)
        save_file(source_path, source)

    target = params["target"]
    target_extension = params["target_extension"]
    target_path = os.path.join(
        temp_dir, os.path.basename(f"target-{formatted_time}.{target_extension}")
    )

    save_file(target_path, target)

    output_path = os.path.join(temp_dir, os.path.basename(f"output.{target_extension}"))

    state_manager.set_item("source_paths", source_paths)
    state_manager.set_item("target_path", target_path)
    state_manager.set_item("output_path", output_path)

    output_image_resolution = detect_image_resolution(target_path)
    output_image_resolutions = create_image_resolutions(output_image_resolution)
    state_manager.set_item(
        "output_image_resolution", pack_resolution(output_image_resolution)
    )

    conditional_process()

    try:
        response = requests.post(
            "http://localhost:3033/scale",
            json={
                #
                "local_path": output_path,
                "output_dir": temp_dir,
            },
        )
        response.raise_for_status()
        res = response.json()["data"]
        local_path = res["local_path"]

        # output_upscale_base64 = to_base64_str(local_path)
        print(local_path)

        return {
            "status": 1,
            "data": {
                # "output": output_upscale_base64,
                "local_path": local_path
            },
        }
    except Exception as e:
        return {"status": 1, "data": {"output": to_base64_str(output_path)}}


def launch():
    app.include_router(router)
    uvicorn.run(app, host=HOST, port=PORT)
