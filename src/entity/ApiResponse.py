
from pydantic import BaseModel
from typing import Optional, TypeVar
import datetime

T = TypeVar("T")

class Response(BaseModel):
    code: int = 200
    message: str = "success"
    data: Optional[T] = None
    timestamp: str = datetime.datetime.now().isoformat()

