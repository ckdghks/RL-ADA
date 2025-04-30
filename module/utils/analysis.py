import os
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob

def plot_miou_trends(csv_path, save_path=None, title=None):
    """
    PPO 또는 GRPO 로그 CSV에서 ΔmIoU 트렌드를 시각화합니다.
    """
    df = pd.read_csv(csv_path)

    if "delta_mIoU" not in df.columns:
        print("❌ 'delta_mIoU' 컬럼이 없습니다.")
        return

    if "episode" in df.columns:
        x = df["episode"]
    elif "step_index" in df.columns:
        x = df["step_index"]
    else:
        x = range(len(df))

    y = df["delta_mIoU"]

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, label="ΔmIoU", color='blue', marker='o')
    plt.xlabel("Episode / Step")
    plt.ylabel("ΔmIoU (fine-tune - pretrained)")
    plt.grid(True)
    plt.title(title if title else "ΔmIoU over time")
    plt.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"✅ Plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def compare_csvs(ppo_csv, grpo_csv, save_path=None):
    """
    PPO와 GRPO의 ΔmIoU 로그를 하나의 그래프에 비교하여 출력합니다.
    """
    df_ppo = pd.read_csv(ppo_csv)
    df_grpo = pd.read_csv(grpo_csv)

    x_ppo = df_ppo["step_index"] if "step_index" in df_ppo else range(len(df_ppo))
    y_ppo = df_ppo["delta_mIoU"]

    x_grpo = df_grpo["episode"] if "episode" in df_grpo else range(len(df_grpo))
    y_grpo = df_grpo["delta_mIoU"]

    plt.figure(figsize=(10, 6))
    plt.plot(x_ppo, y_ppo, label="PPO ΔmIoU", color='green')
    plt.plot(x_grpo, y_grpo, label="GRPO ΔmIoU", color='red')
    plt.xlabel("Episode / Step")
    plt.ylabel("ΔmIoU")
    plt.title("PPO vs GRPO ΔmIoU Comparison")
    plt.grid(True)
    plt.legend()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path)
        print(f"✅ 비교 그래프 저장됨: {save_path}")
    else:
        plt.show()
    plt.close()


def tensorboard_dir_summary(log_root):
    """
    Tensorboard 로그 경로에 존재하는 모든 로그 디렉토리 출력.
    """
    log_dirs = sorted(glob(os.path.join(log_root, "*")))
    print("📊 TensorBoard 로그 디렉토리 목록:")
    for path in log_dirs:
        print(" -", path)
    print("\n📌 실행: tensorboard --logdir", log_root)


# ✅ CLI 실행 지원
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ΔmIoU 로그 분석 및 시각화 도구")

    parser.add_argument('--mode', choices=['plot', 'compare', 'tensorboard'], required=True, help="실행 모드 선택")
    parser.add_argument('--csv', type=str, help="단일 로그 CSV 파일 경로 (plot 모드)")
    parser.add_argument('--ppo_csv', type=str, help="PPO 로그 CSV 경로 (compare 모드)")
    parser.add_argument('--grpo_csv', type=str, help="GRPO 로그 CSV 경로 (compare 모드)")
    parser.add_argument('--save', type=str, default=None, help="시각화 결과 저장 경로")
    parser.add_argument('--title', type=str, default=None, help="플롯 타이틀")
    parser.add_argument('--logdir', type=str, default="tensorboard_logs", help="TensorBoard 로그 디렉토리")

    args = parser.parse_args()

    if args.mode == "plot":
        if not args.csv:
            print("❌ --csv 경로를 지정해주세요.")
        else:
            plot_miou_trends(csv_path=args.csv, save_path=args.save, title=args.title)

    elif args.mode == "compare":
        if not args.ppo_csv or not args.grpo_csv:
            print("❌ --ppo_csv 와 --grpo_csv 경로를 모두 지정해주세요.")
        else:
            compare_csvs(ppo_csv=args.ppo_csv, grpo_csv=args.grpo_csv, save_path=args.save)

    elif args.mode == "tensorboard":
        tensorboard_dir_summary(log_root=args.logdir)
