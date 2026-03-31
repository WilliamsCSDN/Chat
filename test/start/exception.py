if __name__ == '__main__':
    try:
        print(1/0)
    except Exception as e:
        print("{}".format(e))
    except ZeroDivisionError:
        print("不能除0")