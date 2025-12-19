import numpy as np
import tensorflow as tf
from vpg_quadrotor import QuadrotorEnv, actor_creator


if __name__ == "__main__":
    env = QuadrotorEnv(render_mode="human") # create env with rendering ON
    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.n

    actor = actor_creator(obs_size, act_size)   # load trained actor
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
