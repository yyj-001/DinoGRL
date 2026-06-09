import os
from tqdm import tqdm

def to_4_digits_str(number):
    return f"{number:04d}"

def create_directory_if_not_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

root_folder = '/data1/ls/data/HITSZ-VCM'
Train_path = os.path.join(root_folder,'Train_text')
id_folder_list = os.listdir(Train_path)
# s = to_4_digits_str(config.s)
# e = to_4_digits_str(config,e)
# save = ['0058','0099','0107','0319','0390','0432']
for id in tqdm(range(1,503)):
    id = to_4_digits_str(id)
    if id not in id_folder_list:
        print(id)
        continue
    # if id in save:
    #     continue
    id_path = os.path.join(Train_path,id)
    for modal in os.listdir(id_path):
        modal_path = os.path.join(id_path,modal)
        for camera in os.listdir(modal_path):
            camera_path = os.path.join(modal_path,camera)
            for img in os.listdir(camera_path):
                img_path = os.path.join(camera_path,img)
                text_path = img_path.replace('Train_text','Train_text2')
                with open(img_path, 'r') as file:
                    content = file.read()
                    if 'He' in content:
                        content = content.split('He appears to')
                    elif 'She' in content:
                        content = content.split('She appears to')
                    new_content = content[0]
                # text_path = text_path.replace('.jpg','.txt')
                dir_text_path = os.path.dirname(text_path)
                create_directory_if_not_exists(dir_text_path)
                with open(text_path, "w") as file:
                    file.write(new_content)