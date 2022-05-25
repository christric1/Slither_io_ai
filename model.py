from tensorflow.keras.models import *
from tensorflow.keras.layers import *
import tensorflow as tf


def make_actor():
    observation_input = Input(shape=(84, 84, 4))
    length_input = Input(shape=(1,))

    # Block1
    x = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer=tf.random_normal_initializer) \
        (observation_input)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block2
    x = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer=tf.random_normal_initializer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block3
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=tf.random_normal_initializer)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=tf.random_normal_initializer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block4
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=tf.random_normal_initializer)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=tf.random_normal_initializer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block5
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=tf.random_normal_initializer)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=tf.random_normal_initializer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Flatten()(x)

    y = Dense(8, activation='relu')(length_input)
    x = Concatenate()([x, y])

    x = Dense(200, activation='relu')(x)
    x = Dense(200, activation='relu')(x)
    action = Dense(2, activation='tanh')(x)

    actor = Model([observation_input, length_input], action, name="Actor")

    return actor


def make_critic():
    observation_input = Input(shape=(84, 84, 4))
    length_input = Input(shape=(1,))
    action_input = Input(shape=(2,))

    # Block1
    x = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer=tf.random_normal_initializer) \
        (observation_input)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block2
    x = Conv2D(16, (3, 3), activation='relu', padding='same', kernel_initializer=tf.random_normal_initializer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block3
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=tf.random_normal_initializer)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=tf.random_normal_initializer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block4
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=tf.random_normal_initializer)(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=tf.random_normal_initializer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    # Block5
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=tf.random_normal_initializer)(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer=tf.random_normal_initializer)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2))(x)

    x = Flatten()(x)

    x = Concatenate()([x, length_input, action_input])

    x = Dense(200, activation='relu')(x)
    x = Dense(200, activation='relu')(x)
    Q = Dense(1)(x)

    critic = Model([observation_input, length_input, action_input], Q, name="Critc")

    return critic
