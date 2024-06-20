import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

def create_folder(folder_path):
  if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# 儲存圖片到指定資料夾
def save_image(tensor, folder, filename):
  # 轉換回PIL圖片
  image = transforms.ToPILImage()(tensor)
  # 創建資料夾
  create_folder(folder)
  # 儲存圖片
  image.save(os.path.join(folder, filename))

def save_images(file_p):
  file = open(file_p)
  lines = file.readlines()
  image_paths, labels = [], []
  for i in tqdm(range(len(lines)), desc="Saving images"):
    path, label = lines[i].split(' ')
    img_full_path = os.path.join('/homes/nfs/TylerC/workspace/DL/Lab2/', path)
    image = Image.open(img_full_path).convert('RGB')
    if image.mode == 'RGB':
      image = transform(image)

      label_folder_3c = os.path.join('/homes/nfs/TylerC/workspace/DL/Lab2/image_proceeing', f"{os.path.basename(file_p).split('.')[0]}", '3c', str(label))
      label_folder_2c = os.path.join('/homes/nfs/TylerC/workspace/DL/Lab2/image_proceeing', f"{os.path.basename(file_p).split('.')[0]}", '2c', str(label))
      label_folder_1c = os.path.join('/homes/nfs/TylerC/workspace/DL/Lab2/image_proceeing', f"{os.path.basename(file_p).split('.')[0]}", '1c', str(label))

      save_image(image, label_folder_3c, f"{os.path.basename(img_full_path).split('.')[0]}.png")
      save_image(image[[0, 1], :, :], label_folder_2c, f"{os.path.basename(img_full_path).split('.')[0]}_RG.png")
      save_image(image[[1, 2], :, :], label_folder_2c, f"{os.path.basename(img_full_path).split('.')[0]}_GB.png")
      save_image(image[[0, 2], :, :], label_folder_2c, f"{os.path.basename(img_full_path).split('.')[0]}_RB.png")
      save_image(image[0, :, :].unsqueeze(0), label_folder_1c, f"{os.path.basename(img_full_path).split('.')[0]}_R.png")
      save_image(image[1, :, :].unsqueeze(0), label_folder_1c, f"{os.path.basename(img_full_path).split('.')[0]}_G.png")
      save_image(image[2, :, :].unsqueeze(0), label_folder_1c, f"{os.path.basename(img_full_path).split('.')[0]}_B.png")
      
    else:
      print(f"Failed to load image: {img_full_path}")

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

save_images('/homes/nfs/TylerC/workspace/DL/Lab2/train.txt')
save_images('/homes/nfs/TylerC/workspace/DL/Lab2/val.txt')
save_images('/homes/nfs/TylerC/workspace/DL/Lab2/test.txt')
