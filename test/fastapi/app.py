from datetime import datetime
from typing import Optional

import uvicorn
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, APIRouter, WebSocket
from fastapi.responses import JSONResponse


app = FastAPI()

user_router = APIRouter(prefix="/users", tags=["用户"])

@app.get("/test")
async def get_test():
    return "hello world"

@app.get("/test2")
async def get_test2(
        name: str = "",
        age: int = 0
):
    return {"name": name, "age": age}

class User(BaseModel):
    name: str
    age: Optional[int] = None

class UserOut(BaseModel):
    name: str
    age: Optional[int] = None
    create_at: datetime

@app.post("/test3")
async def get_test3(user: User):
    return UserOut(**user.dict(), create_at=datetime.now())

@app.post("/test4")
async def get_test4(user: User):
    return JSONResponse(
        content= {"name":user.name, "age1":user.age}
    )

@user_router.post("/upload")
async def upload(file: UploadFile):
    str = await file.read()
    print(str)
    return (f"f"
            f"ile:{file.filename}")

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    while True:
        data = await websocket.receive_text()
        await websocket.send(f"收到信息：{data}")


app.include_router(user_router)

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)