from __future__ import annotations
from pydantic import BaseModel, Field, field_validator, model_validator

class Student(BaseModel):
    id: int
    age: int = Field(ge=8, le=60)

    @field_validator('age')
    @classmethod
    def checkage(cls, a: int) -> int:
        if a == 8:
            raise ValueError("错误")
        if a ==9:
            return a+1
        return a


    @model_validator(mode='after')
    def checkid(self) -> Student:
        if self.id==666:
            raise ValueError("不能是这个值")
        return self

if __name__ == '__main__':
    u = Student(id=666, age=9)
    print(u.age)
