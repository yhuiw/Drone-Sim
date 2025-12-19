import numpy as np
import tensorflow as tf

# import your environment and actor creator
from vpg_quadrotor import QuadrotorEnv, actor_creator


def visualize_policy():
    # create env with rendering ON
    env = QuadrotorEnv(render_mode="human")
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.n

    # load trained actor
    actor = actor_creator(obs_size, act_size)
    actor.load_weights("vpg_quadrotor_actor.h5")

    # reset env
    obs, _ = env.reset()

    done = False
    while not done:
        # policy inference
        obs_tensor = tf.convert_to_tensor(obs[None], dtype=tf.float32)
        logits = actor(obs_tensor, training=False)
        action = np.argmax(tf.nn.softmax(logits)[0].numpy())

        # step env (this triggers _render_frame internally)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    env.close()


if __name__ == "__main__":
    visualize_policy()
