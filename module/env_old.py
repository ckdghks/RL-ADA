import torch
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from torch.utils.data import DataLoader, Subset
from .dataloader.dataset import CityscapesGTA5
from .core.segmentation_trainer import PretrainedSegModel, fine_tune_seg_model, evaluate_miou
import os
import csv
import matplotlib.pyplot as plt
from PIL import Image
import time
from collections import defaultdict


class CityscapesGTA5Env(gym.Env):
    def __init__(self, root, train_datalist, val_datalist, batch_size=8, selected_ratio=0.05,
                 seg_model_name='deeplabv3plus_resnet101', seg_num_classes=19, checkpoint_path=None, gpu_id=0,
                 log_dir="logs", update_pretrained_if_improved=True, min_finetune_samples=5):
        super(CityscapesGTA5Env, self).__init__()

        self.class_names = [
            'Road', 'Sidewalk', 'Building', 'Wall', 'Fence',
            'Pole', 'Traffic Light', 'Traffic Sign', 'Vegetation',
            'Terrain', 'Sky', 'Person', 'Rider', 'Car',
            'Truck', 'Bus', 'Train', 'Motorcycle', 'Bicycle'
        ]
        self.num_classes = seg_num_classes
        self.gpu_id = gpu_id
        self.update_pretrained_if_improved = update_pretrained_if_improved
        self.min_finetune_samples = min_finetune_samples
        self.start_time = time.time()
        self.episode_start_time = None

        self.train_dataset = CityscapesGTA5(root=root, datalist=train_datalist, split='train')
        self.val_dataset = CityscapesGTA5(root=root, datalist=val_datalist, split='val')
        self.total_samples = len(self.train_dataset)
        self.select_limit = int(self.total_samples * selected_ratio)

        self.selected_indices = []
        self.current_index = 0

        self.batch_size = batch_size
        self.selected_ratio = selected_ratio

        self.action_space = spaces.MultiBinary(2)
        self.observation_space = spaces.Box(low=0, high=255, shape=(3, 1024, 2048), dtype=np.uint8)

        self.pretrained_model = PretrainedSegModel(model_name=seg_model_name, num_classes=seg_num_classes,
                                                   checkpoint_path=checkpoint_path, gpu_id=gpu_id)

        self.log_dir = os.path.join(log_dir, "ppo")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, "ppo_seg_log.csv")
        self.plot_dir = os.path.join(self.log_dir, "plots")
        self.image_dir = os.path.join(self.log_dir, "selected_samples")
        os.makedirs(self.plot_dir, exist_ok=True)

        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'selected_count', 'pretrained_mIoU', 'finetuned_mIoU', 'delta_mIoU',
                'episode_time_hms', 'total_time_hms', 'shaped_reward'
            ])

        self.episode_count = 0
        self.miou_logs = []

    def reset(self, seed=None, options=None):
        print(f"\n🔁 Resetting PPO Environment - Episode {self.episode_count}")
        self.episode_start_time = time.time()
        self.current_index = 0
        self.selected_indices.clear()
        self.episode_count += 1
        return self._get_observation(), {}

    def _get_observation(self):
        if self.current_index >= len(self.train_dataset):
            return np.zeros((3, 1024, 2048), dtype=np.uint8)
        sample = self.train_dataset[self.current_index]
        image = sample["images"]
        return image.numpy()

    def step(self, action):
        select_flag, trigger_flag = action
        reward = 0

        if select_flag == 1:
            self.selected_indices.append(self.current_index)
            # debug
            # sample = self.train_dataset[self.current_index]
            # img_np = sample["images"].numpy().transpose(1, 2, 0).astype(np.uint8)
            # save_path = os.path.join(self.image_dir, f"ep{self.episode_count}_idx{self.current_index}.png")
            # Image.fromarray(img_np).save(save_path)

        if trigger_flag == 1:
            if len(self.selected_indices) >= self.min_finetune_samples:
                reward = self._fine_tune_and_evaluate()
            else:
                print(f"⚠️ Not enough samples to fine-tune: {len(self.selected_indices)} / {self.min_finetune_samples}")
                reward = -0.1

        self.current_index += 1
        done = (
            self.current_index >= len(self.train_dataset) or
            len(self.selected_indices) >= self.select_limit
        )

        return self._get_observation(), reward, done, False, {}

    def _fine_tune_and_evaluate(self):
        print(f"\n🧮 Step {self.current_index} | 🔧 Fine-tuning with {len(self.selected_indices)} / {self.select_limit} samples...")
        selected_dataset = torch.utils.data.Subset(self.train_dataset, self.selected_indices)
        selected_dataloader = DataLoader(selected_dataset, batch_size=4, shuffle=True)

        fine_tuned_model = fine_tune_seg_model(self.pretrained_model, selected_dataloader, gpu_id=self.gpu_id)

        pretrained_miou, _ = evaluate_miou(self.pretrained_model, self.val_dataset, self.num_classes,
                                           ignore_label=255, class_names=self.class_names, gpu_id=self.gpu_id, save_vis=True,
                                           vis_dir=f"{self.log_dir}/pre_vis")

        finetuned_miou, _ = evaluate_miou(fine_tuned_model, self.val_dataset, self.num_classes,
                                          ignore_label=255, class_names=self.class_names, gpu_id=self.gpu_id, save_vis=True,
                                          vis_dir=f"{self.log_dir}/ft_vis")

        delta_miou = finetuned_miou - pretrained_miou
        sample_ratio = len(self.selected_indices) / self.select_limit
        alpha = 0.5
        shaped_reward = delta_miou * (1 + alpha * sample_ratio)
        print(f"📈 ΔmIoU: {delta_miou:.4f} → Shaped Reward: {shaped_reward:.4f} | (Fine-tuned: {finetuned_miou:.4f}, Pretrained: {pretrained_miou:.4f})")

        if self.update_pretrained_if_improved and finetuned_miou > pretrained_miou:
            print("✅ mIoU improved! Updating pretrained model.")
            self.pretrained_model = fine_tuned_model
        else:
            print("📉 No improvement in mIoU. Keeping previous pretrained model.")

        episode_seconds = time.time() - self.episode_start_time
        total_seconds = time.time() - self.start_time

        ep_mins, ep_secs = divmod(episode_seconds, 60)
        ep_hours, ep_mins = divmod(ep_mins, 60)
        total_mins, total_secs = divmod(total_seconds, 60)
        total_hours, total_mins = divmod(total_mins, 60)

        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.episode_count, len(self.selected_indices), pretrained_miou, finetuned_miou, delta_miou,
                f"{int(ep_hours)}h{int(ep_mins)}m{int(ep_secs)}s",
                f"{int(total_hours)}h{int(total_mins)}m{int(total_secs)}s",
                shaped_reward
            ])

        self.miou_logs.append((
            self.episode_count, self.current_index, pretrained_miou, finetuned_miou, delta_miou, shaped_reward, len(self.selected_indices)
        ))
        self._plot_logs()
        self._plot_episode_graph()
        return shaped_reward

    def _plot_logs(self):
        if not self.miou_logs:
            return

        # 에피소드별로 묶기
        episode_dict = defaultdict(list)
        for log in self.miou_logs:
            episode = log[0]
            episode_dict[episode].append(log)

        for episode, logs in episode_dict.items():
            logs.sort(key=lambda x: x[1])  # step 기준 정렬

            steps, pretrained, finetuned, delta, shaped_rewards, selected_counts = zip(
                *[(s, p, f, d, r, c) for _, s, p, f, d, r, c in logs]
            )

            fig, ax1 = plt.subplots(figsize=(12, 7))

            # 🎯 주요 성능 지표 (왼쪽 y축)
            ax1.plot(steps, pretrained, marker='o', label='Pretrained mIoU')
            ax1.plot(steps, finetuned, marker='o', label='Finetuned mIoU')
            ax1.plot(steps, delta, marker='x', linestyle='--', label='Δ mIoU')
            ax1.plot(steps, shaped_rewards, marker='s', linestyle='-.', label='Shaped Reward')
            ax1.set_xlabel("Step")
            ax1.set_ylabel("Score (mIoU / Reward)")
            ax1.tick_params(axis='y')
            ax1.grid(True)

            # 🎯 샘플링 개수 (오른쪽 y축)
            ax2 = ax1.twinx()
            ax2.plot(steps, selected_counts, marker='^', color='tab:red', label='Selected Count')
            ax2.axhline(y=self.select_limit, color='r', linestyle='--', alpha=0.3, label=f"Limit = {self.select_limit}")
            ax2.set_ylabel("Selected Samples")

            # 🎯 범례 병합
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax2.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')

            plt.title(f"Episode {episode}: Segmentation Performance & Sampling Count by Step")
            fig.tight_layout()
            filename = os.path.join(self.plot_dir, f"episode_{episode}_miou_by_step.png")
            plt.savefig(filename)
            plt.close()

    def _plot_episode_graph(self):
        if not self.miou_logs:
            return

        # metric 별로 에피소드 기준 분리
        ep_to_reward = defaultdict(list)
        ep_to_delta = defaultdict(list)

        for ep_id, _, _, _, delta, reward, _ in self.miou_logs:
            ep_to_reward[ep_id].append(reward)
            ep_to_delta[ep_id].append(delta)

        sorted_eps = sorted(ep_to_reward.keys())
        reward_data = [ep_to_reward[ep] for ep in sorted_eps]
        delta_data = [ep_to_delta[ep] for ep in sorted_eps]

        num_eps = len(sorted_eps)
        x = np.arange(num_eps)
        width = 0.35  # reward와 delta 간 간격

        fig, ax = plt.subplots(figsize=(14, 6))

        # reward box
        bp1 = ax.boxplot(reward_data, positions=x - width / 2, widths=0.3, patch_artist=True,
                        boxprops=dict(facecolor="lightgreen"), showmeans=True, medianprops=dict(color='green'),
                        meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='green'))

        # delta box
        bp2 = ax.boxplot(delta_data, positions=x + width / 2, widths=0.3, patch_artist=True,
                        boxprops=dict(facecolor="lightblue"), showmeans=True, medianprops=dict(color='blue'),
                        meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='blue'))

        ax.set_xticks(x)
        ax.set_xticklabels(sorted_eps)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Score")
        ax.set_title("Reward & ΔmIoU Distribution per Episode")
        ax.grid(True)

        # 범례 수동 생성
        custom_lines = [
            plt.Line2D([0], [0], color='lightgreen', lw=10),
            plt.Line2D([0], [0], color='lightblue', lw=10)
        ]
        ax.legend(custom_lines, ['Shaped Reward', 'Δ mIoU'], loc='upper right')

        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, "boxplot_reward_and_delta_per_episode.png"))
        plt.close()
            
    def render(self, mode="human"):
        pass

    def close(self):
        pass

'''
GRPO 알고리즘 적용 env
'''
import torch.nn as nn
import torch.nn.functional as F

class MLPSelectorPolicy(torch.nn.Module):
    def __init__(self, input_dim=3*1024*2048, hidden_dim=512, num_outputs=4):
        super().__init__()
        self.num_outputs = num_outputs
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, 2)
        )

    def forward(self, x):
        return [self.net(x) for _ in range(self.num_outputs)]

class CityscapesGTA5GRPOEnv(gym.Env):
    def __init__(self, root, train_datalist, val_datalist,
                 seg_model_name='deeplabv3plus_resnet101', seg_num_classes=19,
                 checkpoint_path=None, selected_ratio=0.03,
                 batch_size=1, gpu_id=0, log_dir='logs',
                 input_shape=(3, 1024, 2048),
                 update_pretrained_if_improved=True,
                 min_finetune_samples=5,
                 num_action_heads=4,
                 finetune_interval=5):  # 추가된 인자

        super().__init__()

        self.min_finetune_samples = min_finetune_samples
        self.gpu_id = gpu_id
        self.seg_num_classes = seg_num_classes
        self.selected_ratio = selected_ratio
        self.batch_size = batch_size
        self.input_shape = input_shape
        self.update_pretrained_if_improved = update_pretrained_if_improved
        self.num_action_heads = num_action_heads
        self.finetune_interval = finetune_interval  # 초기화

        self.log_dir = os.path.join(log_dir, "grpo")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_path = os.path.join(self.log_dir, "grpo_episode_log.csv")

        self.start_time = time.time()
        self.episode_start_time = None

        self.train_dataset = CityscapesGTA5(root=root, datalist=train_datalist, split='train')
        self.val_dataset = CityscapesGTA5(root=root, datalist=val_datalist, split='val')
        self.total_samples = len(self.train_dataset)
        self.select_limit = int(self.total_samples * self.selected_ratio)
        self.indices = list(range(self.total_samples))

        self.action_space = spaces.MultiBinary(2)
        self.observation_space = spaces.Box(low=0, high=255, shape=self.input_shape, dtype=np.uint8)

        self.pretrained_model = PretrainedSegModel(model_name=seg_model_name,
                                                   num_classes=seg_num_classes,
                                                   checkpoint_path=checkpoint_path,
                                                   gpu_id=gpu_id)

        self.episode_count = 0
        self.miou_logs = []
        self.current_index = 0
        self.selected_indices = set()  # 중복 방지 위해 set 사용

        self._init_csv_log()

    def _init_csv_log(self):
        with open(self.log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                'episode', 'selected_count', 'pre_mIoU', 'ft_mIoU', 'delta_mIoU',
                'episode_time_hms', 'total_time_hms', 'shaped_reward'
            ])

    def _step_internal(self, action):
        select_flag, trigger_flag = action
        reward = 0

        if select_flag == 1:
            self.selected_indices.add(self.indices[self.current_index])

        if (
            trigger_flag == 1 or
            self.current_index % self.finetune_interval == 0 or
            self.current_index == self.total_samples - 1
        ):
            if len(self.selected_indices) >= self.min_finetune_samples:
                reward = self._evaluate_selection()
            else:
                reward = -0.1
        else:
            reward = -0.1

        self.current_index += 1
        return reward

    def get_observation(self):
        if self.current_index >= len(self.train_dataset):
            return np.zeros(self.input_shape, dtype=np.uint8)
        sample = self.train_dataset[self.indices[self.current_index]]
        return sample["images"].numpy()

    def reset(self, seed=None, options=None):
        print(f"\n🔁 Resetting GRPO Environment - Episode {self.episode_count + 1}")
        self.episode_start_time = time.time()
        self.current_index = 0
        self.selected_indices.clear()
        self.episode_count += 1
        return self.get_observation(), {}

    def _evaluate_selection(self):
        print(f"\n🧮 Step {self.current_index} | 🔧 Fine-tuning with {len(self.selected_indices)} / {self.select_limit} samples...")
        loader = DataLoader(Subset(self.train_dataset, list(self.selected_indices)), batch_size=1, shuffle=True)
        torch.cuda.empty_cache()
        fine_tuned_model = fine_tune_seg_model(self.pretrained_model, loader, gpu_id=self.gpu_id)

        with torch.no_grad():
            pre_miou, _ = evaluate_miou(self.pretrained_model, self.val_dataset, self.seg_num_classes,
                                        gpu_id=self.gpu_id, save_vis=True,
                                        vis_dir=os.path.join(self.log_dir, "pre_vis_grpo"))

            ft_miou, _ = evaluate_miou(fine_tuned_model, self.val_dataset, self.seg_num_classes,
                                       gpu_id=self.gpu_id, save_vis=True,
                                       vis_dir=os.path.join(self.log_dir, "ft_vis_grpo"))

        delta = ft_miou - pre_miou
        sample_ratio = len(self.selected_indices) / self.select_limit
        alpha = 0.5
        shaped_reward = delta * (1 + alpha * sample_ratio)

        if self.update_pretrained_if_improved and ft_miou > pre_miou:
            print("✅ mIoU improved! Updating pretrained model.")
            del self.pretrained_model
            torch.cuda.empty_cache()
            self.pretrained_model = fine_tuned_model
        else:
            del fine_tuned_model
            torch.cuda.empty_cache()

        episode_seconds = time.time() - self.episode_start_time
        total_seconds = time.time() - self.start_time
        ep_mins, ep_secs = divmod(episode_seconds, 60)
        ep_hours, ep_mins = divmod(ep_mins, 60)
        total_mins, total_secs = divmod(total_seconds, 60)
        total_hours, total_mins = divmod(total_mins, 60)

        with open(self.log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                self.episode_count, len(self.selected_indices),
                pre_miou, ft_miou, delta,
                f"{int(ep_hours)}h{int(ep_mins)}m{int(ep_secs)}s",
                f"{int(total_hours)}h{int(total_mins)}m{int(total_secs)}s",
                shaped_reward
            ])

        self.miou_logs.append((
            self.episode_count, self.current_index, pre_miou, ft_miou, delta, shaped_reward, len(self.selected_indices)
        ))

        self._plot_episode_graph()

        return shaped_reward

    def _plot_episode_graph(self):
        if not self.miou_logs:
            return

        ep_to_reward = {}
        ep_to_delta = {}
        ep_to_selected = {}

        for ep_id, _, _, _, delta, reward, selected_count in self.miou_logs:
            ep_to_reward[ep_id] = reward
            ep_to_delta[ep_id] = delta
            ep_to_selected[ep_id] = selected_count

        sorted_eps = sorted(ep_to_reward.keys())
        rewards = [ep_to_reward[ep] for ep in sorted_eps]
        deltas = [ep_to_delta[ep] for ep in sorted_eps]
        selected_counts = [ep_to_selected[ep] for ep in sorted_eps]

        x = np.arange(len(sorted_eps))

        fig, ax1 = plt.subplots(figsize=(14, 6))

        ax1.plot(x, rewards, marker='o', label='Shaped Reward', color='green')
        ax1.plot(x, deltas, marker='x', linestyle='--', label='Δ mIoU', color='blue')
        ax1.set_xlabel("Episode")
        ax1.set_ylabel("Score (mIoU / Reward)")
        ax1.set_xticks(x)
        ax1.set_xticklabels(sorted_eps)
        ax1.grid(True)

        ax2 = ax1.twinx()
        ax2.plot(x, selected_counts, marker='^', linestyle='-.', label='Selected Count', color='red')
        ax2.set_ylabel("Selected Samples")

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

        plt.title("Reward & ΔmIoU per Episode with Selected Samples")
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "lineplot_reward_and_delta_per_episode_grpo.png"))
        plt.close()
