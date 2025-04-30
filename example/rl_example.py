import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

# 1️⃣ 환경(Environment) 생성
env = gym.make("CartPole-v1")  # 예제 환경

# 2️⃣ PPO 모델 초기화 (CPU에서 학습)
model = PPO("MlpPolicy", env, verbose=1, device="cpu")  # CPU 강제 사용

# 3️⃣ PPO 학습
print("🚀 Training PPO Model on CPU...")
model.learn(total_timesteps=100_000)

# 4️⃣ 모델 저장
model.save("ppo_cartpole_cpu")

# 5️⃣ 평가 함수 실행
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print(f"📈 Evaluation: Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

# 6️⃣ 모델 로드 및 사용 (CPU에서 실행)
loaded_model = PPO.load("ppo_cartpole_cpu", device="cpu")
obs, _ = env.reset()

for _ in range(1000):
    action
