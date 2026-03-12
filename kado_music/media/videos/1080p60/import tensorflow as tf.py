import tensorflow as tf
from tensorflow.keras import layers, Model
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

class VAE(Model):
    def __init__(self, latent_dim=2):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Энкодер
        self.encoder = tf.keras.Sequential([
            layers.Input(shape=(28, 28, 1)),
            layers.Flatten(),
            layers.Dense(512, activation='relu'),
            layers.Dense(256, activation='relu'),
            layers.Dense(latent_dim * 2)  # mean и log variance
        ])
        
        # Декодер
        self.decoder = tf.keras.Sequential([
            layers.Input(shape=(latent_dim,)),
            layers.Dense(256, activation='relu'),
            layers.Dense(512, activation='relu'),
            layers.Dense(784, activation='sigmoid'),
            layers.Reshape((28, 28, 1))
        ])
    
    def encode(self, x):
        mean_logvar = self.encoder(x)
        mean, logvar = tf.split(mean_logvar, num_or_size_splits=2, axis=1)
        return mean, logvar
    
    def reparameterize(self, mean, logvar):
        eps = tf.random.normal(shape=mean.shape)
        return eps * tf.exp(logvar * 0.5) + mean
    
    def decode(self, z):
        return self.decoder(z)
    
    def call(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        reconstructed = self.decode(z)
        return reconstructed, mean, logvar

# Функция потерь для VAE
def vae_loss(x, reconstructed, mean, logvar):
    reconstruction_loss = tf.reduce_mean(
        tf.reduce_sum(
            tf.keras.losses.binary_crossentropy(x, reconstructed),
            axis=(1, 2)
        )
    )
    kl_loss = -0.5 * tf.reduce_mean(
        tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar), axis=1)
    )
    return reconstruction_loss + kl_loss

# Загрузка данных
(x_train, _), (x_test, _) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)

# Создание и обучение модели
vae = VAE(latent_dim=2)
vae.compile(optimizer='adam', loss=vae_loss)

# Обучение
history = vae.fit(
    x_train, x_train,
    epochs=30,
    batch_size=128,
    validation_split=0.2
)

# Генерация новых изображений
def generate_images(vae, num_images=10):
    random_latent_vectors = tf.random.normal(shape=(num_images, vae.latent_dim))
    generated_images = vae.decode(random_latent_vectors)
    
    fig, axes = plt.subplots(1, num_images, figsize=(20, 4))
    for i in range(num_images):
        axes[i].imshow(generated_images[i, :, :, 0], cmap='gray')
        axes[i].axis('off')
    plt.show()

generate_images(vae)