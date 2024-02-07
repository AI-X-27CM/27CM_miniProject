import face_recognition
from fastapi import FastAPI, Form, Request, Depends,status, UploadFile, File
import cv2
from fastapi.staticfiles import StaticFiles
import numpy as np
from starlette.templating import Jinja2Templates
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from database import SessionLocal, engine
import models 
from sqlalchemy.orm import Session
from starlette.responses import RedirectResponse
import os
import glob ## 추가
import io
from fastapi.responses import JSONResponse


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
    # os.remove(models.User.userid)
    user = db.query(models.User).filter(models.User.user_id == user_id).first()
    os.remove(user.user_image)
    db.delete(user)
    db.commit()
    url = app.url_path_for("home")
    return RedirectResponse(url=url, status_code=status.HTTP_303_SEE_OTHER)

def compare_faces(known_face_encodings, face_encoding_to_check, tolerance=0.25): ## 추가
    return list(face_recognition.face_distance(known_face_encodings, face_encoding_to_check) <= tolerance)

@app.get("/testify") ## 추가
async def testify(req: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse("testify_test.html", { "request": req})

@app.post("/verify_image/") #추가
async def verify_image(image: UploadFile = File(...)):
    db = SessionLocal()
    all_images = db.query(models.User).all()
    file_list = [user.user_image for user in all_images]
    print(file_list)

    try:
        # 클라이언트에서 전송한 이미지의 얼굴 인코딩 계산
        image_data = await image.read()
        img = face_recognition.load_image_file(io.BytesIO(image_data))
        face_encoding_to_check = face_recognition.face_encodings(img)[0]

        # 데이터베이스 이미지들의 얼굴 인코딩 계산
        known_face_encodings = []
        for file in file_list:
            image_path = file
            image = face_recognition.load_image_file(image_path)
            face_encoding = face_recognition.face_encodings(image)[0]
            known_face_encodings.append(face_encoding)

        # 얼굴 유사도 비교
        results = compare_faces(known_face_encodings, face_encoding_to_check)
        if any(results):
            return JSONResponse(content={"result": "PASS"})
        else:
            return JSONResponse(content={"result": "Not PASS"})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
