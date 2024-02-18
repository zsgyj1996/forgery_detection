import os

file_list_path = '/data/shuozhang/dataset/casia/file_list2.txt'
tp_directory = '/data/shuozhang/dataset/casia/Gt'

# 读取文件列表
with open(file_list_path, 'r') as file:
    file_list = file.read().splitlines()

# 删除文件
for file_name in file_list:
    file_path = os.path.join(tp_directory, file_name)
    if os.path.exists(file_path):
        os.remove(file_path)
        print(f"Deleted: {file_path}")
    else:
        print(f"Not found: {file_path}")
