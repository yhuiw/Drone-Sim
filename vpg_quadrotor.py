"""ELEN E6601 Project, small drone control sim
Author @AlexWei
Last modified: 12/16/2025

ACADEMIC INTEGRITY STATEMENT:
adapted from my earlier CartPole project, discount_reward() and sample_traj() functions are direct copy;
state traj plotting codes by GenAI
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium import spaces
from tqdm import tqdm

tf.config.set_visible_devices([], 'GPU')    # CPU faster for such small network


class QuadrotorEnv(gym.Env):
    """custom Gymnasium environment for planar quadrotor hover control
    state:
        y    : horizontal position (m)
        y_dot: horizontal velocity (m/s)
        φ    : pitch angle (rad)
        φ_dot: pitch angular velocity (rad/s)
    
    action (discrete): 0 = neg torque, 1 = 0 torque, 2 = pos torque
    action (continuous): torque in [-τ_max, τ_max]
    
    reward: +1 for each step the drone stays within bounds
    termination: |φ| > 15° or |y| > 1.0 m"""
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}
    
    def __init__(self, render_mode=None, continuous=False):
        super().__init__()
        
        # our small drone physical params
        self.m_total = 0.536        # total mass, kg
        self.g = 9.81               # m/s^2
        self.L_arm = 0.1422         # arm length, m
        self.I_yy = 0.0015          # moment of inertia, kg*m^2

        self.tau_max = 0.05         # restricted max torque, N*m (from motor differential)
        
        # state limits for termination
        self.y_max = 1.0                # max horizontal displacement, m
        self.phi_max = 15 * np.pi / 180 # max pitch angle, rad (15°)
        
        # simulation params
        self.dt = 0.02              # timestep, s (50 Hz control)
        self.max_steps = 500        # max episode length
        
        self.continuous = continuous
        
        # define action space
        if continuous:
            self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        else:
            # discrete: 0=negative torque, 1=zero, 2=positive torque
            self.action_space = spaces.Discrete(3)
        
        # obsv space (generous bounds for learning)
        high = np.array([2.0, 5.0, np.pi / 4, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        self.state = None
        self.steps = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # initialize near hover with small random perturbation
        self.state = self.np_random.uniform(low=-0.05, high=0.05, size=(4,))
        self.state = self.state.astype(np.float32)
        self.steps = 0
        
        if self.render_mode == 'human':
            self._render_frame()
        
        return self.state, {}
    
    def step(self, action):
        y, y_dot, phi, phi_dot = self.state
        if self.continuous: # convert action to torque
            tau = float(action[0]) * self.tau_max
        else:
            # discrete: -1, 0, +1 scaled by tau_max
            tau = (action - 1) * self.tau_max
        
        # drone dynamics (Euler integration)
        # phi_ddot = tau / I_yy, y_ddot = g * sin(phi) ≈ g * phi for small ang
        phi_ddot = tau / self.I_yy
        y_ddot = self.g * np.sin(phi)  # use full nonlinear for RL
        
        # update state
        phi_new = phi + phi_dot * self.dt
        phi_dot_new = phi_dot + phi_ddot * self.dt
        y_new = y + y_dot * self.dt
        y_dot_new = y_dot + y_ddot * self.dt
        
        self.state = np.array([y_new, y_dot_new, phi_new, phi_dot_new], dtype=np.float32)
        self.steps += 1
        
        # check termination
        terminated = bool(abs(phi_new) > self.phi_max or abs(y_new) > self.y_max)
        truncated = self.steps >= self.max_steps
        
        # reward +1 for staying in bounds, bonus for being close to hover
        if not terminated:
            reward = 1.0
            # Small bonus for staying centered
            reward += 0.1 * (1 - abs(phi_new) / self.phi_max)
            reward += 0.1 * (1 - abs(y_new) / self.y_max)
        else:
            reward = 0.0
        
        if self.render_mode == 'human':
            self._render_frame()
        
        return self.state, reward, terminated, truncated, {}
    
    def _render_frame(self):
        try:
            import pygame
        except ImportError:
            return
        
        screen_width = 600
        screen_height = 400
        if self.screen is None:
            pygame.init()
            if self.render_mode == 'human':
                pygame.display.set_caption('Quadrotor Hover')
                self.screen = pygame.display.set_mode((screen_width, screen_height))
            self.clock = pygame.time.Clock()
        
        # colors
        WHITE = (255, 255, 255)
        BLACK = (0, 0, 0)
        BLUE = (0, 100, 200)
        RED = (200, 50, 50)
        GREEN = (50, 200, 50)
        
        self.screen.fill(WHITE)
        
        if self.state is None:
            return
        
        y, y_dot, phi, phi_dot = self.state
        
        # scale and center
        scale = 200  # pixels per meter
        center_x = screen_width // 2
        center_y = screen_height // 2
        
        # drone body dimensions (scaled)
        body_width = int(0.2 * scale)  # ~20cm wingspan shown
        body_height = int(0.03 * scale)
        
        # drone position in pixels
        drone_x = center_x + int(y * scale)
        drone_y = center_y  # altitude fixed in 2D model
        
        # draw reference lines
        pygame.draw.line(self.screen, (200, 200, 200), (0, center_y), (screen_width, center_y), 1)
        pygame.draw.line(self.screen, (200, 200, 200), (center_x, 0), (center_x, screen_height), 1)
        
        # draw bounds
        pygame.draw.line(self.screen, RED, (center_x - int(self.y_max * scale), 0),
                        (center_x - int(self.y_max * scale), screen_height), 2)
        pygame.draw.line(self.screen, RED, (center_x + int(self.y_max * scale), 0),
                        (center_x + int(self.y_max * scale), screen_height), 2)
        
        # draw drone body as rotated rectangle
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        
        # corners of drone body (before rotation)
        hw, hh = body_width // 2, body_height // 2
        corners = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
        
        # rotate and translate
        rotated_corners = []
        for cx, cy in corners:
            rx = cx * cos_phi - cy * sin_phi + drone_x
            ry = cx * sin_phi + cy * cos_phi + drone_y
            rotated_corners.append((rx, ry))
        
        pygame.draw.polygon(self.screen, BLUE, rotated_corners)
        
        # draw propeller indicators (circles at ends)
        prop_offset = body_width // 2
        left_prop = (drone_x - prop_offset * cos_phi, drone_y - prop_offset * sin_phi)
        right_prop = (drone_x + prop_offset * cos_phi, drone_y + prop_offset * sin_phi)
        pygame.draw.circle(self.screen, GREEN, (int(left_prop[0]), int(left_prop[1])), 8)
        pygame.draw.circle(self.screen, GREEN, (int(right_prop[0]), int(right_prop[1])), 8)
        
        # draw center marker
        pygame.draw.circle(self.screen, RED, (drone_x, drone_y), 3)
        
        # display state info
        font = pygame.font.Font(None, 24)
        info_text = f'y={y:.3f}m  φ={phi*180/np.pi:.1f}°  step={self.steps}'
        text_surface = font.render(info_text, True, BLACK)
        self.screen.blit(text_surface, (10, 10))
        
        pygame.display.flip()
        self.clock.tick(self.metadata['render_fps'])
    
    def close(self):
        if self.screen is not None:
            import pygame
            pygame.quit()
            self.screen = None


# register custom environment
gym.register(id='Quadrotor-v0', entry_point='__main__:QuadrotorEnv', max_episode_steps=500)


def actor_creator(state_dim, action_dim):
    """Create actor network for policy gradient"""
    state_input = layers.Input(shape=(state_dim,))
    hidden1 = layers.Dense(64, activation='relu')(state_input)
    hidden2 = layers.Dense(64, activation='relu')(hidden1)
    logits = layers.Dense(action_dim)(hidden2)
    return models.Model(inputs=state_input, outputs=logits)


def critic_creator(state_dim):
    """Create critic network for value estimation"""
    state_input = layers.Input(shape=(state_dim,))
    hidden1 = layers.Dense(64, activation='relu')(state_input)
    hidden2 = layers.Dense(64, activation='relu')(hidden1)
    value_output = layers.Dense(1, activation=None)(hidden2)
    return models.Model(inputs=state_input, outputs=value_output)


def sample_traj(mdl, env, batch=2000, seed=None):
    """Sample trajectories from environment using current policy"""
    s, a, r, not_dones = [], [], [], []
    curr_reward_list = []
    collected = 0
    env_seed = seed
    act_size = env.action_space.n

    while collected < batch:
        state, _ = env.reset(seed=env_seed)
        if env_seed is not None:
            env_seed = None
        curr_reward = 0
        terminated, truncated = False, False

        while not (terminated or truncated):
            state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32)
            logits = mdl(state_tensor, training=False)
            probs = tf.nn.softmax(logits).numpy()[0]
            action = np.random.choice(act_size, p=probs)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            s.append(state)
            a.append(action)
            r.append(reward)
            not_dones.append(0.0 if done else 1.0)

            state = next_state
            curr_reward += reward
            collected += 1

            if done:
                break

        curr_reward_list.append(curr_reward)

    return (np.array(s, dtype=np.float32), np.array(a, dtype=np.int32),
            np.array(r, dtype=np.float32), np.array(not_dones, dtype=np.float32),
            np.mean(curr_reward_list))


def discount_rewards(reward_buffer, dones, gamma):
    """Compute discounted returns"""
    g_t = np.zeros_like(reward_buffer, dtype=float)
    running_add = 0
    num_traj = 0
    for t in reversed(range(len(reward_buffer))):
        running_add = reward_buffer[t] + gamma * running_add * dones[t]
        g_t[t] = running_add
        if dones[t] == 0:
            num_traj += 1
    if len(dones) > 0 and dones[-1] != 0:
        num_traj += 1
    if len(reward_buffer) == 0:
        num_traj = 0
    return g_t.astype(np.float32), max(1, num_traj)


def train(model_actor, model_critic, opt_actor, opt_critic, s, a, r, dones, gamma):
    """Train actor & critic networks"""
    s = tf.convert_to_tensor(s, dtype=tf.float32)
    a = tf.convert_to_tensor(a, dtype=tf.int32)
    r = tf.convert_to_tensor(r, dtype=tf.float32)
    dones = tf.convert_to_tensor(dones, dtype=tf.float32)

    g_t, _ = discount_rewards(r.numpy(), dones.numpy(), gamma)
    g_t = tf.convert_to_tensor(g_t, dtype=tf.float32)

    # train critic
    with tf.GradientTape() as tape:
        critics = model_critic(s, training=True)
        critics = tf.squeeze(critics, axis=1)
        loss_critic = tf.keras.losses.mean_squared_error(g_t, critics)
    critic_grads = tape.gradient(loss_critic, model_critic.trainable_variables)
    opt_critic.apply_gradients(zip(critic_grads, model_critic.trainable_variables))

    # train actor
    with tf.GradientTape() as tape:
        logits = model_actor(s, training=True)
        log_prob = -tf.nn.sparse_softmax_cross_entropy_with_logits(labels=a, logits=logits)
        advantages = g_t - tf.stop_gradient(tf.squeeze(model_critic(s, training=False), axis=1))
        advantages = (advantages - tf.reduce_mean(advantages)) / (tf.math.reduce_std(advantages) + 1e-8)
        loss_actor = -tf.reduce_mean(log_prob * advantages)
    actor_grads = tape.gradient(loss_actor, model_actor.trainable_variables)
    opt_actor.apply_gradients(zip(actor_grads, model_actor.trainable_variables))

    return loss_critic.numpy(), loss_actor.numpy()


def main():
    env = QuadrotorEnv()
    obs_size = env.observation_space.shape[0]  # 4: y, y_dot, phi, phi_dot
    act_size = env.action_space.n  # 3: negative/zero/positive torque

    print(f"state space: {obs_size} dimensions")
    print(f"action space: {act_size} discrete actions")
    print(f"drone mass: {env.m_total} kg")
    print(f"max torque: {env.tau_max:.1f} N·m")
    
    # hyperparams
    GAMMA = 0.99
    last_n_reward = 100
    TRAIN_EPISODES = 1500
    actor_lr = 3e-4
    critic_lr = 1e-3
    batch_size = 5000
    target_reward = 550  # success threshold (slightly above max 500 due to bonuses)

    # initialize model & optimizers
    actor = actor_creator(obs_size, act_size)
    critic = critic_creator(obs_size)
    actor_optimizer = tf.keras.optimizers.Adam(learning_rate=actor_lr)
    critic_optimizer = tf.keras.optimizers.Adam(learning_rate=critic_lr)
    
    episode_reward_history = []
    running_rewards, episode_rewards = [], []
    actor_losses, critic_losses = [], []
    consistency = 0


    pbar = tqdm(range(TRAIN_EPISODES))
    for ep in pbar: # main training loop
        states, actions, rewards, n_dones, episode_reward = sample_traj(actor, env, batch=batch_size)
        critic_loss, actor_loss = train(actor, critic, actor_optimizer, critic_optimizer, states, actions, rewards, n_dones, GAMMA)
        actor_losses.append(actor_loss)
        critic_losses.append(critic_loss)
        episode_reward_history.append(episode_reward)
        episode_rewards.append(episode_reward)

        if len(episode_reward_history) > last_n_reward:
            del episode_reward_history[0]
        running_reward = np.mean(episode_reward_history)
        running_rewards.append(running_reward)

        pbar.set_postfix(EpReward=f'{episode_reward:.1f}', RunReward=f'{running_reward:.1f}')

        if running_reward >= target_reward:
            consistency += 1
            if consistency >= 10:
                print(f"\nearly stopping at ep {ep}")
                break
        else:
            consistency = 0

    pbar.close()
    env.close()
    actor.save_weights("vpg_quadrotor_actor.h5")    # save weights
    critic.save_weights("vpg_quadrotor_critic.h5")

    # plot training progress
    plt.figure(figsize=(12, 8))
    plt.subplot(2, 1, 1)
    plt.plot(running_rewards, label="running reward", linewidth=2)
    plt.plot(episode_rewards, label="ep reward", alpha=0.4)
    plt.axhline(y=target_reward, linestyle='--', label='target')
    plt.xlabel("ep")
    plt.ylabel("reward")
    plt.title("Quadrotor RL Training - Reward Evolution")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(actor_losses, label="actor loss", alpha=0.7)
    plt.plot(critic_losses, label="critic loss", alpha=0.7)
    plt.xlabel("ep")
    plt.ylabel("loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return actor, critic


def test_policy(actor, num_episodes=5, render=True):
    render_mode = 'human' if render else None
    env = QuadrotorEnv(render_mode=render_mode)
    
    test_rewards = []
    test_lengths = []
    for ep in range(num_episodes):
        state, _ = env.reset()
        total_reward = 0
        steps = 0
        done = False
        
        states_history = [state.copy()]
        while not done:
            state_tensor = tf.convert_to_tensor(np.expand_dims(state, axis=0), dtype=tf.float32)
            logits = actor(state_tensor, training=False)
            # deterministic action selection for testing
            action = np.argmax(tf.nn.softmax(logits).numpy()[0])
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
            states_history.append(state.copy())
        
        test_rewards.append(total_reward)
        test_lengths.append(steps)
        print(f"ep {ep + 1}: reward = {total_reward:.1f}, steps = {steps}")
        if ep == num_episodes - 1:  # plot state trajectory for last episode
            states_history = np.array(states_history)
            t = np.arange(len(states_history)) * env.dt
            
            plt.figure(figsize=(10, 8))
            plt.subplot(4, 1, 1)
            plt.plot(t, states_history[:, 2] * 180 / np.pi, 'b-', linewidth=1.5)
            plt.ylabel('φ (deg)')
            plt.title('Quadrotor State Trajectory')
            plt.grid(True)
            
            plt.subplot(4, 1, 2)
            plt.plot(t, states_history[:, 3] * 180 / np.pi, 'b-', linewidth=1.5)
            plt.ylabel('φ̇ (deg/s)')
            plt.grid(True)
            
            plt.subplot(4, 1, 3)
            plt.plot(t, states_history[:, 0], 'b-', linewidth=1.5)
            plt.ylabel('y (m)')
            plt.grid(True)
            
            plt.subplot(4, 1, 4)
            plt.plot(t, states_history[:, 1], 'b-', linewidth=1.5)
            plt.ylabel('ẏ (m/s)')
            plt.xlabel('time (s)')
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
    
    env.close()

    print(f"mean reward: {np.mean(test_rewards):.1f} ± {np.std(test_rewards):.1f}")
    print(f"mean ep length: {np.mean(test_lengths):.0f} steps")


if __name__ == "__main__":  # train agent and test policy
    actor, critic = main()
    test_policy(actor, num_episodes=3, render=True)
