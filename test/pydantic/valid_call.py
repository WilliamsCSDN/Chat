from pydantic import BaseModel, Field, validate_call


class Student(BaseModel):
    id: int
    age: int = Field(ge=8, le=60)


@validate_call
def checkid(id: int, age: int):
     if id==666:
        raise ValueError("不能是这个值")

if __name__ == '__main__':
    checkid("sdaf",123)
    print(1)
