from gfpgan import GFPGANer
from time import sleep, time
from facefusion.core import conditional_process
import facefusion.globals as globals
import uvicorn
from fastapi import FastAPI, APIRouter, Body
import os
import base64
import tempfile
import base64
import cv2
from basicsr.utils import imwrite
from facefusion.vision import read_static_image
from facefusion.face_analyser import get_many_faces
from PIL import Image


# Define a function to load .env file
def load_env_file(filepath):
    with open(filepath) as f:
        for line in f:
            # Ignore comments and empty lines
            if line.startswith('#') or not line.strip():
                continue
            # Split key and value
            key, value = line.strip().split('=', 1)
            os.environ[key] = value


# Load the .env file
load_env_file('.env')

app = FastAPI()
router = APIRouter()

arch = 'clean'
channel_multiplier = 2
model_name = 'GFPGANv1.4'
url = 'https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth'
bg_upsampler = None
# determine model paths
model_path = os.path.join('experiments/pretrained_models', model_name + '.pth')
if not os.path.isfile(model_path):
    model_path = os.path.join('gfpgan/weights', model_name + '.pth')
if not os.path.isfile(model_path):
    # download pre-trained models from url
    model_path = url


def isFile(path):
    return os.path.splitext(path)[1] != ""


def update_global_variables(params):
    for var_name, value in params.items():
        if value is not None:
            if hasattr(globals, var_name):
                setattr(globals, var_name, value)


def to_base64_str(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
        return encoded_string.decode('utf-8')


def save_file(file_path: str, encoded_data: str):
    decoded_data = base64.b64decode(encoded_data)
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(file_path, "wb") as file:
        file.write(decoded_data)


def apply_args():
    from facefusion.vision import is_image, is_video, detect_image_resolution, detect_video_resolution, detect_video_fps, create_image_resolutions, create_video_resolutions, pack_resolution
    from facefusion.normalizer import normalize_fps
    if is_image(globals.target_path):
        output_image_resolution = detect_image_resolution(globals.target_path)
        output_image_resolutions = create_image_resolutions(output_image_resolution)
        if globals.output_image_resolution in output_image_resolutions:
            globals.output_image_resolution = globals.output_image_resolution
        else:
            globals.output_image_resolution = pack_resolution(output_image_resolution)
    if is_video(globals.target_path):
            output_video_resolution = detect_video_resolution(globals.target_path)
            output_video_resolutions = create_video_resolutions(output_video_resolution)
            if globals.output_video_resolution in output_video_resolutions:
                globals.output_video_resolution = globals.output_video_resolution
            else:
                globals.output_video_resolution = pack_resolution(output_video_resolution)
    if globals.output_video_fps or is_video(globals.target_path):
        globals.output_video_fps = normalize_fps(globals.output_video_fps) or detect_video_fps(globals.target_path)


def upscaleImg(img_path:str, ext:str, output_path:str ):
    start_time_upscale = time()

    img_name = os.path.basename(img_path)
    print(f'Processing {img_name} ...')
    basename, ext = os.path.splitext(img_name)
    input_img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        
    restorer = GFPGANer(
        model_path=model_path,
        upscale=1,
        arch=arch,
        channel_multiplier=channel_multiplier,
        bg_upsampler=bg_upsampler)

    # restore faces and background if necessary
    cropped_faces, restored_faces, restored_img = restorer.enhance(
        input_img,
        has_aligned=False,
        only_center_face=False,
        paste_back=True,
        weight=None)

    extension = ext

    # save restored img
    if restored_img is not None:

        # save_restore_path = os.path.join(args.output, f'{basename}.{extension}')
        # Check if args.output is a directory
        if os.path.isdir(output_path):
            save_restore_path = os.path.join(output_path, f'{basename}.{extension}')
        # Check if args.output is a file
        elif isFile(output_path):
            save_restore_path = output_path
        # If args.output is neither a file nor a directory, handle the error or create a new directory/file
        else:
            # Handle this situation as you see fit (e.g., raise an error, create a directory, etc.)
            raise ValueError("args.output is neither a valid file nor a directory path")

        imwrite(restored_img, save_restore_path)

    if(isFile(output_path)):
        print(f'Results: {output_path}')
    else:
        print(f'Results are in the [{output_path}] folder.')

    seconds = '{:.2f}'.format((time() - start_time_upscale) % 60)
    print(f'Upscale in: [{seconds}] seconds.')

    return


def get_location_frames(reference_frame):
    faces = get_many_faces(reference_frame)
    face = faces[0]

    # for face in faces:
    start_x, start_y, end_x, end_y = map(int, face.bounding_box)
    padding_x = int((end_x - start_x) * 0.25)
    padding_y = int((end_y - start_y) * 0.25)
    start_x = max(0, start_x - padding_x)
    start_y = max(0, start_y - padding_y)
    end_x = max(0, end_x + padding_x)
    end_y = max(0, end_y + padding_y)
    crop_frame = [start_x, start_y, end_x, end_y]

    return crop_frame


def crop_image_by_location(image_path: str, crop_frame, cropFaceSourcePath: str) -> None:
    image = Image.open(image_path)
    width, height = image.size

    (start_x, start_y, end_x, end_y) = map(int, crop_frame)
    if end_x > width:
        end_x = width
    if end_y > height:
        end_y = height

    cropped_image = image.crop((start_x, start_y, end_x, end_y))
    # cropped_image.save(cropFaceSourcePath)
    cropped_image_path = os.path.join( tempfile.mkdtemp(), os.path.basename(f'source-face-small.png'))
    cropped_image.save(cropped_image_path)
    upscaleImg(cropped_image_path, "png", cropFaceSourcePath)


def get_face_process(source_path) -> None:
	return get_location_frames(read_static_image(source_path))


@router.post("/")
async def process_frames(params = Body(...)) -> dict:
    update_global_variables(params)
    sources = params['sources']
    source_extension = params['source_extension']
    source_paths = []

    tempDir = tempfile.mkdtemp()

    for i, source in enumerate(sources):
        source_path = os.path.join(tempDir, os.path.basename(f'source{i}.{source_extension}'))
        save_file(source_path, source)
        source_paths.append(source_path)

    globals.face_analyser_order="best-worst"
    firstFace = get_face_process(source_paths[0])

    cropFaceSourcePath = os.path.join(tempDir, os.path.basename(f'source-face.png'))
    crop_image_by_location(source_paths[0], firstFace, cropFaceSourcePath)

    print(f'cropFaceSourcePath :>> {cropFaceSourcePath}')

    target = params['target']
    target_extension = params['target_extension']
    target_path = os.path.join(tempDir, os.path.basename(f'target.{target_extension}'))
    save_file(target_path, target)
    globals.source_paths = source_paths
    globals.target_path = target_path
    globals.output_path = os.path.join(tempDir, os.path.basename(f'output.{target_extension}'))
    apply_args()
    print(globals.source_paths)
    print(globals.target_path)
    print(globals.output_path)
    conditional_process()
    # output = to_base64_str(globals.output_path)
    output_path = os.path.join(tempDir, os.path.basename(f'output-upscale.{target_extension}'))

    upscaleImg(globals.output_path, target_extension, output_path)
    
    output_upscale_base64 = to_base64_str(output_path) 
    return {"output": output_upscale_base64}


port = int(os.getenv('PORT', 3000))


def launch():
    app.include_router(router)
    uvicorn.run(app, host="0.0.0.0", port=port)
