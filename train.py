import wandb
from tensorflow.keras.optimizers import Adam
import os
from env import Slitherio
from model import *
from utils import *

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

wandb.init(project="slither", entity="christee")
config = wandb.config

# hyperparameter
config.max_memory_size = 1000000
config.mini_batch_size = 16
config.gamma = 0.99
config.tau = 0.001
config.std_dev = 0.2
config.actor_lr = 0.0001
config.critic_lr = 0.001

noise_generator = OUActionNoise(mean=np.zeros(1), std_deviation=float(config.std_dev) * np.ones(1))

# create model
actor = make_actor()
critic = make_critic()
actor_target = make_actor()
critic_target = make_critic()

actor.summary()
critic.summary()

# sync model
actor_target.set_weights(actor.get_weights())
critic_target.set_weights(critic.get_weights())

# Adam optimizer
actor_optimizer = Adam(config.actor_lr)
critic_optimizer = Adam(config.critic_lr)

replay_memory = ReplayBuffer(config.max_memory_size)


def main():

    try:
        # load model
        actor.load_weights('./models/actor.h5')
        critic.load_weights('./models/critic.h5')
        actor_target.load_weights('./models/actor_target.h5')
        critic_target.load_weights('./models/critic_target.h5')
    except:
        pass

    # Get model parameter
    actor_vars = actor.trainable_variables
    actor_target_vars = actor_target.trainable_variables
    critic_vars = critic.trainable_variables
    critic_target_vars = critic_target.trainable_variables

    # Create env
    env = Slitherio(nickname="Bot_AI")
    env = FrameStack(env)
    env.start()

    # train record
    updates = 0
    episodes = 0
    highest_score = 0
    longest_duration = 0

    while True:
        observation = env.reset()
        length = env.score
        terminal = False
        mean_actor_loss = 0
        mean_critic_loss = 0
        mean_total_loss = 0
        timeStep = 0

        while not terminal:
            # Get action
            sampled_actions = tf.squeeze(actor([tf.expand_dims(observation, axis=0), tf.expand_dims(length, axis=0)]))
            noise = noise_generator()
            action = sampled_actions.numpy() + noise

            angle, acceleration = action
            acceleration = int(acceleration > 0)
            next_observation, reward, terminal, info = env.step([angle, acceleration])
            next_length = env.score
            replay_memory.store(observation, length, action, reward, next_observation, next_length)
            observation = next_observation
            length = next_length
            states, lengths, actions, rewards, next_states, next_lengths = replay_memory.sample(config.mini_batch_size)

            with tf.GradientTape() as tape:
                target_actions = actor_target([next_states, next_lengths], training=True)
                target_values = rewards + config.gamma * critic_target([next_states, next_lengths, target_actions],
                                                                       training=True)
                pred_values = critic([states, lengths, actions], training=True)
                critic_loss = tf.math.reduce_mean(tf.math.square(target_values - pred_values))

            critic_grads = tape.gradient(critic_loss, critic_vars)
            critic_optimizer.apply_gradients(zip(critic_grads, critic_vars))

            with tf.GradientTape() as tape:
                actions = actor([states, lengths], training=True)
                pred_values = critic([states, lengths, actions], training=True)
                actor_loss = -tf.math.reduce_mean(pred_values)

            actor_grads = tape.gradient(actor_loss, actor_vars)
            actor_optimizer.apply_gradients(zip(actor_grads, actor_vars))

            total_loss = actor_loss + critic_loss

            update_target(actor_target_vars, actor_vars, config.tau)
            update_target(critic_target_vars, critic_vars, config.tau)

            timeStep += 1
            updates += 1

            mean_critic_loss += (critic_loss - mean_critic_loss) / timeStep
            mean_actor_loss += (actor_loss - mean_actor_loss) / timeStep
            mean_total_loss += (total_loss - mean_total_loss) / timeStep

            wandb.log({
                'actor_loss': actor_loss,
                'critic_loss': critic_loss,
                'total_loss': total_loss
            })

        episodes += 1
        score = env.score
        highest_score = max(highest_score, score)
        longest_duration = max(longest_duration, timeStep)

        print("""
        Episode:          %d
        Score:            %d
        High Score:       %d
        Updates:          %d
        Duration:         %d
        Longest Survival: %d
        Actor Loss:       %.6f
        Critic Loss:      %.6f
        Total Loss:       %.6f
        """ % (episodes, score, highest_score, updates, timeStep, longest_duration, mean_actor_loss, mean_critic_loss,
               mean_total_loss))

        wandb.log({
            'score': score,
            'highest_score': highest_score,
            'updates': updates,
            'duration': timeStep,
            'longest_duration': longest_duration,
            'mean_actor_loss': mean_actor_loss,
            'mean_critic_loss': mean_critic_loss,
            'mean_total_loss': mean_total_loss
        })

        # save model
        actor.save_weights(f'./models/actor.h5')
        critic.save_weights(f'./models/critic.h5')
        actor_target.save_weights(f'./models/actor_target.h5')
        critic_target.save_weights(f'./models/critic_target.h5')

        if episodes > 50:
            break


if __name__ == '__main__':
    while True:
        main()
