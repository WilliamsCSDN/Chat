from pymilvus import connections, utility


connections.connect(
    alias="default",
    host="localhost",
    port="19530"
)

if __name__ == '__main__':
    print(utility.get_server_version())  # v2.4.x
