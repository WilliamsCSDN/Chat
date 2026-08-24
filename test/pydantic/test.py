from datetime import date

from pydantic import BaseModel, Field


class User(BaseModel):
    id: int = Field(strict=True)
    name: str | None = None
    create_time: date


if __name__ == '__main__':
    u = User(id=666, name="Williams", create_time=date(2026,8,24))
    print(u)
