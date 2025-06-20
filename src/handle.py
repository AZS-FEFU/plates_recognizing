import shutil
from uuid import uuid4
from fastapi import APIRouter, UploadFile
from settings import settings

from .script import main

handle_router = APIRouter()

@handle_router.post("/recognize_plate")
async def recognize_plate(file: UploadFile):
    """
    Uploads a file and returns its filename and content type.
    """
    file_splitted = file.filename.split(".")

    if len(file_splitted) <= 1:
        return dict()

    filename = f"{str(uuid4())}.{file_splitted[-1]}"
    file_location = f"{settings.input_path}/{filename}" # Define where to save the file
    file_output = f"{settings.output_path}/{filename}" # Define where to save the file
    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    plate_number = main(input_path=file_location, output_path=file_output, yolo_path=settings.yolo_path)
    
    return {"plate_number": plate_number}