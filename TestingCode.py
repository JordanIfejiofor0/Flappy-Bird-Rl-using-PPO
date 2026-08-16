import torch
import numpy as np
from rlgame import FlappyBirdEnv
from model import ActorCrtic


MODEL_PATH = "output/FlappyBirdProject/1e4-1000Train.pth"

scores = []
def test():
    env = FlappyBirdEnv()

    model = ActorCrtic()
    model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
    model.eval()

    print("Loaded trained model 1000 version!")
    print("-" * 50)

    for episode in range(50):

        obs, _ = env.reset()

        done = False
        truncated = False
        score = 0

        while not (done or truncated):

            env.render()

            obs_tensor = torch.FloatTensor(obs)

            with torch.no_grad():
                action, _, _ = model.get_action(
                    obs_tensor,
                    deterministic=True
                )

            obs, reward, done, truncated, _ = env.step(
                action.item()
            )

            score += reward
            scores.append(score)

        print(f"Episode {episode + 1}: Score = {score}, Pipes = {env.score}")

    env.close()


if __name__ == "__main__":
    test()
    print("\n===== Results =====")
    print(f"Mean:   {np.mean(scores):.2f}")
    print(f"Std:    {np.std(scores):.2f}")
    print(f"Median: {np.median(scores):.2f}")
    print(f"Min:    {np.min(scores):.2f}")
    print(f"Max:    {np.max(scores):.2f}")
