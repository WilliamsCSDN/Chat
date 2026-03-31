class Student:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    def get_user_info(self):
        print(f"{self.name} {self.age}")
        return "sadf"


if __name__ == '__main__':
    student = Student("williams",28)
    res = student.get_user_info()

    print(res)