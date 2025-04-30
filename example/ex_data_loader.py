import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader.dataset import CityscapesGTA5, CityscapesUnlabeled  # 데이터셋 클래스 import

'''
기본 데이터 로더 확인 코드 
'''

# 데이터셋 경로 및 리스트 파일 설정
# root_path = "/workspace/mount/SSD_4T_b/datasets/cityscapes"  # GTA5 또는 Cityscapes 데이터 경로
root_path = "/workspace/mount/SSD_4T_b/datasets/gta5/data"  # GTA5 또는 Cityscapes 데이터 경로
# datalist_file = "/workspace/research_2025/RLADA/dataloader/init_data/cityscapes/train.txt"  # 이미지 & 라벨 리스트 파일
datalist_file = "/workspace/research_2025/RLADA/dataloader/init_data/GTA5/train.txt"  # 이미지 & 라벨 리스트 파일

# 데이터셋 로드
dataset = CityscapesGTA5(root=root_path, datalist=datalist_file, split="train")
dataloader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)

# 첫 번째 배치 가져오기
batch = next(iter(dataloader))

# 데이터 확인 (파일로 저장)
images, labels = batch["images"], batch["labels"]
fig, axes = plt.subplots(2, 4, figsize=(12, 6))

for i in range(4):
    axes[0, i].imshow(images[i].permute(1, 2, 0))  # 이미지 출력
    axes[0, i].set_title("Input Image")
    axes[0, i].axis("off")

    axes[1, i].imshow(dataset.decode_target(labels[i].numpy()))  # 레이블 출력
    axes[1, i].set_title("Segmentation Label")
    axes[1, i].axis("off")

# 저장 (파일명: cityscapes_gta5_sample.png)
plt.savefig("gta5_sample.png", bbox_inches="tight", dpi=300)
plt.close()