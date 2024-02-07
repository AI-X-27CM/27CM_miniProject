import face_recognition
from fastapi import FastAPI, Form, Request, Depends,status, UploadFile, File
import cv2
from fastapi.staticfiles import StaticFiles
import numpy as np
from insightface.app import FaceAnalysis
from starlette.templating import Jinja2Templates
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from database import SessionLocal, engine
import models 
from sqlalchemy.orm import Session
from starlette.responses import RedirectResponse
import os


models.Base.metadata.create_all(bind=engine)
templates = Jinja2Templates(directory="templates")

app = FastAPI()
UPLOAD_DIRECTORY = "static/image"

abs_path = os.path.dirname(os.path.realpath(__file__))
app.mount("/static", StaticFiles(directory=f"{abs_path}/static"))

def get_db():
    db = SessionLocal()
    try: 
        yield db
    finally:
        db.close()

@app.get("/")
async def home(req: Request, db: Session = Depends(get_db)):
    users = db.query(models.User).all()
    return templates.TemplateResponse("index.html", { "request": req, "users": users })

@app.post("/add")
def add_user(req: Request, image: UploadFile = File(...), user_name: str = Form(...), db: Session = Depends(get_db)):
    
    # 업로드된 이미지 파일의 경로
    file_path = os.path.join(UPLOAD_DIRECTORY, user_name + ".jpg")
    with open(file_path, "wb") as file_object:
            file_object.write(image.file.read())

    new_user = models.User(user_name=user_name, user_image="static/image/" + user_name + ".jpg")
    db.add(new_user)
    db.commit()
    url = app.url_path_for("home")
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)

@app.get("/delete/{user_id}")
def add(req: Request, user_id: int, db: Session = Depends(get_db)):
    user = db.query(models.User).filter(models.User.user_id == user_id).first()
    db.delete(user)
    db.commit()                                          
    
    url = app.url_path_for("home")
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)
