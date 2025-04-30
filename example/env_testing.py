from env import CityscapesGTA5Env
from stable_baselines3 import PPO
import torch

# ✅ 경로 설정
city_path = "/workspace/mount/SSD_4T_b/datasets/cityscapes"  # GTA5 또는 Cityscapes 데이터 경로
train_city_list_file = "/workspace/research_2025/RLADA/dataloader/init_data/cityscapes/train.txt"  # 이미지 & 라벨 리스트 파일
val_city_list_file = "/workspace/research_2025/RLADA/dataloader/init_data/cityscapes/val.txt"  # 이미지 & 라벨 리스트 파일

# ✅ 환경 생성
env = CityscapesGTA5Env(
    root=city_path,
    train_datalist=train_city_list_file,
    val_datalist=val_city_list_file,
    batch_size=8,
    selected_ratio=0.05,  # 전체 데이터 중 5%만 선택
    seg_model_name='deeplabv3plus_resnet101',
    seg_num_classes=19,
    checkpoint_path='/workspace/research_2025/D2ADA/deeplabv3plus_resnet101_GTA5_warmup.tar',
    gpu_id=1
)

# ✅ PPO 모델 초기화
model = PPO("MlpPolicy", env, verbose=1, device="cpu")

# ✅ PPO 학습 실행 (작게 실행해보기)
print("🧪 Start PPO training...")
model.learn(total_timesteps=1000)  # 테스트용으로 적은 스텝 실행

# ✅ 학습 후 저장
model.save("ppo_cityscapes_test")

# ✅ 선택된 인덱스 확인
print(f"✅ 선택된 샘플 수: {len(env.sampled_indices)} / {len(env.all_indices)}")

env.close()