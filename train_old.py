import os
import argparse
import torch
import torch.nn.functional as F
from torch.optim import Adam
from stable_baselines3 import PPO
from module.env import CityscapesGTA5Env, CityscapesGTA5GRPOEnv, MLPSelectorPolicy
from module.utils.analysis import (
    plot_miou_trends,
    compare_csvs,
    tensorboard_dir_summary
)
def train_grpo(env, total_episodes=10, hidden_dim=512, lr=3e-4, save_dir="trained_models", device='cpu'):
    input_dim = 3 * 1024 * 2048
    policy = MLPSelectorPolicy(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    for ep in range(total_episodes):
        obs, _ = env.reset()
        done = False

        while not done:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(device)
            obs_tensor = obs_tensor.view(1, -1)  # [1, 3*1024*2048]

            logits_list = policy(obs_tensor)  # list of tensors, each [1, 2]
            probs_list = [torch.sigmoid(logits) for logits in logits_list]  # list of [1, 2]

            log_probs = []
            rewards = []

            for probs in probs_list:
                m = torch.distributions.Bernoulli(probs)
                action_tensor = m.sample()  # [1, 2]
                log_prob = m.log_prob(action_tensor).sum()
                action = action_tensor.squeeze(0).int().tolist()  # → [select_flag, trigger_flag]

                reward = env._step_internal(action)  # ✅ 핵심: 각 action 개별 평가
                log_probs.append(log_prob)
                rewards.append(reward)

                if env.current_index >= len(env.train_dataset) or len(env.selected_indices) >= env.select_limit:
                    done = True
                    break

            # 🧠 상대 보상 기반 pairwise loss 계산
            loss = 0
            n = len(rewards)
            for i in range(n):
                for j in range(n):
                    if i == j:
                        continue
                    diff = rewards[i] - rewards[j]
                    loss += -diff * (log_probs[i] - log_probs[j])
            loss = loss / (n * (n - 1)) if n > 1 else 0

            if loss != 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(f"[GRPO Episode {ep}] Relative loss: {loss.item():.4f}, reward_mean: {np.mean(rewards):.4f}")

            obs = env.get_observation()

    os.makedirs(save_dir, exist_ok=True)
    torch.save(policy.state_dict(), os.path.join(save_dir, "grpo_cityscapes_policy.pt"))
    print("✅ GRPO 학습 완료 및 모델 저장!")

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

    # 🤖 GRPO 설정
    parser.add_argument('--total_episodes', type=int, default=10)
    parser.add_argument('--grpo_hidden_dim', type=int, default=512)

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


    elif args.mode == "grpo":
        print("🚀 GRPO 환경 초기화")
        env = CityscapesGTA5GRPOEnv(
            root=args.data_root,
            train_datalist=args.train_list,
            val_datalist=args.val_list,
            selected_ratio=args.selected_ratio,
            seg_model_name='deeplabv3plus_resnet101',
            seg_num_classes=args.seg_num_classes,
            checkpoint_path=args.checkpoint_path,
            gpu_id=args.gpu_id
        )

        print("🧠 GRPO 학습 시작")
        train_grpo(
            env,
            total_episodes=args.total_episodes,
            hidden_dim=args.grpo_hidden_dim,
            lr=args.learning_rate,
            save_dir=args.save_dir,
            device=f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )

        # ΔmIoU 시각화
        plot_miou_trends(
            csv_path="logs_grpo/grpo_episode_log.csv",
            save_path="plots/grpo_trend.png",
            title="GRPO ΔmIoU"
        )

    elif args.mode == "both":
        print("🚀 [BOTH] PPO 및 GRPO 환경 초기화 및 학습 시작")

        # === PPO ===
        env_ppo = CityscapesGTA5Env(
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
            env=env_ppo,
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

        plot_miou_trends(
            csv_path="logs/ppo_seg_log.csv",
            save_path="plots/ppo_trend.png",
            title="PPO ΔmIoU"
        )

        # === GRPO ===
        env_grpo = CityscapesGTA5GRPOEnv(
            root=args.data_root,
            train_datalist=args.train_list,
            val_datalist=args.val_list,
            selected_ratio=args.selected_ratio,
            seg_model_name='deeplabv3plus_resnet101',
            seg_num_classes=args.seg_num_classes,
            checkpoint_path=args.checkpoint_path,
            gpu_id=args.gpu_id
        )

        print("🧠 GRPO 학습 시작")
        train_grpo(
            env_grpo,
            total_episodes=args.total_episodes,
            hidden_dim=args.grpo_hidden_dim,
            lr=args.learning_rate,
            save_dir=args.save_dir,
            device=f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu"
        )

        plot_miou_trends(
            csv_path="logs/grpo_episode_log.csv",
            save_path="plots/grpo_trend.png",
            title="GRPO ΔmIoU"
        )

        # === 비교 분석 ===
        compare_csvs(
            ppo_csv="logs/ppo_seg_log.csv",
            grpo_csv="logs/grpo_episode_log.csv",
            save_path="plots/ppo_vs_grpo_comparison.png"
        )

    else:
        raise ValueError("❌ mode는 'ppo' 또는 'grpo' 또는 'both' 중 하나여야 합니다.")

if __name__ == "__main__":
    main()
