import os
from fastapi import FastAPI

from src.handle import handle_router

app = FastAPI()

app.include_router(handle_router)

