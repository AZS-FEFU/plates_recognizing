import os
from fastapi import FastAPI

from src.huita import main

app = FastAPI()

yolo_path = "./weights/YOLO/best.pt"
input_path = "./datatest/origin"
output_path = "./datatest/processed"


def get_main(input_path=input_path, output_path = output_path, yolo_path=yolo_path):
    recognized_plates = []
    for image in os.listdir(input_path):
        recognized_plates.append(main(f"{input_path}/{image}", f"{output_path}/{image}", yolo_path))
    return recognized_plates    
if __name__ == "__main__":
    get_main()
