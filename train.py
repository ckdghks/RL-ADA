import os
import argparse
import torch
import torch.nn.functional as F
from torch.optim import Adam
from stable_baselines3 import PPO
from module.env import CityscapesGTA5Env
from module.utils.analysis import (
    plot_miou_trends,
    compare_csvs,
    tensorboard_dir_summary
)


def main():
    parser = argparse.ArgumentParser()

    # 🔧 학습 모드
    parser.add_argument('--mode', choices=['ppo', 'grpo', 'both'], required=True)

    # 📁 데이터 경로 설정
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--train_list', type=str, required=True)
    parser.add_argument('--val_list', type=str, required=True)
    parser.add_argument('--checkpoint_path', type=str, required=True)

    # 💾 저장 경로
    parser.add_argument('--save_dir', type=str, default='trained_models')

    # 🧠 세그멘테이션 설정
    parser.add_argument('--seg_num_classes', type=int, default=19)
    parser.add_argument('--selected_ratio', type=float, default=0.05)

    # ⚙️ PPO 설정
    parser.add_argument('--total_timesteps', type=int, default=10000)
    parser.add_argument('--ppo_steps', type=int, default=64)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=3e-4)

    # 🧩 디바이스
    parser.add_argument('--gpu_id', type=int, default=0)

    args = parser.parse_args()

    if args.mode == "ppo":
        print("🚀 PPO 환경 초기화")
        env = CityscapesGTA5Env(
            root=args.data_root,
            train_datalist=args.train_list,
            val_datalist=args.val_list,
            batch_size=8,
            selected_ratio=args.selected_ratio,
            seg_model_name='deeplabv3plus_resnet101',
            seg_num_classes=args.seg_num_classes,
            checkpoint_path=args.checkpoint_path,
            gpu_id=args.gpu_id
        )

        model = PPO(
            policy="MlpPolicy",
            env=env,
            verbose=1,
            device="cpu",
            batch_size=args.batch_size,
            n_steps=args.ppo_steps,
            learning_rate=args.learning_rate,
        )

        print("🧠 PPO 학습 시작")
        model.learn(total_timesteps=args.total_timesteps)
        os.makedirs(args.save_dir, exist_ok=True)
        model.save(os.path.join(args.save_dir, "ppo_cityscapes_finetune"))
        print("✅ PPO 모델 저장 완료")

        # ΔmIoU 시각화
        plot_miou_trends(
            csv_path="logs/ppo_seg_log.csv",
            save_path="plots/ppo_trend.png",
            title="PPO ΔmIoU"
        )
    else:
        raise ValueError("❌ mode는 'ppo' 또는 'grpo' 또는 'both' 중 하나여야 합니다.")

if __name__ == "__main__":
    main()
