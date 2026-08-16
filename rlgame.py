import random
import sys
import pygame

pygame.init()

# =========================
# Constants
# =========================

SCREEN_WIDTH = 400
SCREEN_HEIGHT = 600
FPS = 60

SKY_BLUE = (135, 206, 235)
YELLOW = (255, 255, 0)
GREEN = (0, 200, 0)
BLACK = (0, 0, 0)

screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("RL Flappy Bird")
clock = pygame.time.Clock()


# =========================
# Environment
# =========================

class FlappyBirdEnv:

    def __init__(self):

        # Bird
        self.bird_x = 80
        self.bird_y = 300
        self.bird_width = 30
        self.bird_height = 24
        self.bird_vel = 0

        # Physics
        self.gravity = 0.4
        self.jump_strength = -7

        # Pipes
        self.pipe_width = 60
        self.pipe_gap = 150
        self.pipe_vel = -3
        self.pipes = []

        # Game state
        self.score = 0
        self.game_over = False

    # =========================
    # Create Pipe
    # =========================

    def create_pipe(self):

        top_height = random.randint(50, 350)

        bottom_height = (
            SCREEN_HEIGHT - top_height - self.pipe_gap
        )

        top_rect = pygame.Rect(
            SCREEN_WIDTH,
            0,
            self.pipe_width,
            top_height
        )

        bottom_rect = pygame.Rect(
            SCREEN_WIDTH,
            SCREEN_HEIGHT - bottom_height,
            self.pipe_width,
            bottom_height
        )

        return {
            "top": top_rect,
            "bottom": bottom_rect,
            "passed": False
        }

    # =========================
    # Observation
    # =========================

    def get_observation(self):

        # Find nearest pipe in front of bird
        nearest_pipe = None

        for pipe in self.pipes:

            if pipe["top"].right >= self.bird_x:
                nearest_pipe = pipe
                break

        # Safety check
        if nearest_pipe is None:
            nearest_pipe = self.pipes[0]

        pipe_distance = (
            nearest_pipe["top"].x - self.bird_x
        )

        gap_y = nearest_pipe["top"].height

        observation = [
            self.bird_y,
            self.bird_vel,
            pipe_distance,
            gap_y
        ]

        return observation

    # =========================
    # Reset
    # =========================

    def reset(self):

        self.bird_y = 300
        self.bird_vel = 0

        self.pipes.clear()

        self.score = 0
        self.game_over = False

        # Initial pipe
        self.pipes.append(self.create_pipe())

        return self.get_observation(), {}

    def close(self):
        pygame.quit()

    # =========================
    # Step
    # =========================

    def step(self, action):

        reward = 0
        terminated = False

        # ---------------------
        # 1. Action
        # ---------------------

        if action == 1:
            self.bird_vel = self.jump_strength

        # ---------------------
        # 2. Bird Physics
        # ---------------------

        self.bird_vel += self.gravity
        self.bird_y += self.bird_vel

        # ---------------------
        # 3. Bird Rect
        # ---------------------

        bird_rect = pygame.Rect(
            self.bird_x,
            int(self.bird_y),
            self.bird_width,
            self.bird_height
        )

        # ---------------------
        # 4. Move Pipes
        # ---------------------

        for pipe in self.pipes:

            pipe["top"].x += self.pipe_vel
            pipe["bottom"].x += self.pipe_vel

        # ---------------------
        # 5. Collision
        # ---------------------

        if (
            self.bird_y <= 0
            or self.bird_y >= SCREEN_HEIGHT - self.bird_height
        ):
            terminated = True

        for pipe in self.pipes:

            if (
                bird_rect.colliderect(pipe["top"])
                or bird_rect.colliderect(pipe["bottom"])
            ):
                terminated = True

        # ---------------------
        # 6. Reward
        # ---------------------

        # Small reward for staying alive
        if not terminated:
            reward = 0.1


        # Reward for passing pipe
        for pipe in self.pipes:

            if (
                not pipe["passed"]
                and pipe["top"].right < self.bird_x
            ):
                reward += 3
                pipe["passed"] = True
                self.score += 1

        # ---------------------
        # 7. Remove old pipes
        # ---------------------

        self.pipes = [
            pipe for pipe in self.pipes
            if pipe["top"].right > 0
        ]

        # ---------------------
        # 8. Spawn new pipe
        # ---------------------

        if len(self.pipes) == 0:

            self.pipes.append(
                self.create_pipe()
            )

        # ---------------------
        # 9. Observation
        # ---------------------

        observation = self.get_observation()

        return observation, reward, terminated, False, {}

    # =========================
    # Render
    # =========================

    def render(self):

        screen.fill(SKY_BLUE)

        # Bird
        pygame.draw.rect(
            screen,
            YELLOW,
            (
                self.bird_x,
                int(self.bird_y),
                self.bird_width,
                self.bird_height
            ),
            border_radius=4
        )

        # Pipes
        for pipe in self.pipes:

            pygame.draw.rect(
                screen,
                GREEN,
                pipe["top"]
            )

            pygame.draw.rect(
                screen,
                GREEN,
                pipe["bottom"]
            )

        # Score
        font = pygame.font.SysFont(None, 36)

        score_surface = font.render(
            f"Score: {self.score}",
            True,
            BLACK
        )

        screen.blit(
            score_surface,
            (10, 10)
        )

        pygame.display.flip()

        clock.tick(FPS)




