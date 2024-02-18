import os

# 读取tp_list.txt文件的绝对路径
tp_list_path = '/data/shuozhang/merged/tp_list.txt'
with open(tp_list_path, 'r') as file:
    tp_list = file.readlines()

# 定义目标目录
target_folder = '/data/shuozhang/merged/dataset/Gt/'

# 逐行处理tp_list.txt中的绝对路径
for tp_path in tp_list:
    # 替换fake为mask，并将文件名后缀替换为.png
    new_path = tp_path.replace('fake', 'mask').rstrip('\n')[:-4] + '.png'

    # 复制替换后的文件到目标目录
    os.makedirs(target_folder, exist_ok=True)
    os.system(f"cp {new_path} {target_folder}")

print("操作完成")
