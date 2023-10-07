import tensorflow as tf
from gan import GAN


def generator_loss(fake_output):
    """Wasserstein loss for the generator"""
    return -tf.reduce_mean(fake_output)


def discriminator_loss(real_output, fake_output):
    """Wasserstein loss for the discriminator"""
    return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)


class WGAN_GP(GAN):
    def __init__(self, generator, discriminator, latent_vector_dim, folder_path,
                 gen_learning_rate=1e-4, disc_learning_rate=1e-4, lambda_gp=10.0):
        super().__init__(generator, discriminator, latent_vector_dim, folder_path,
                         gen_learning_rate, disc_learning_rate)
        self.lambda_gp = lambda_gp

    def gradient_penalty(self, real_images, generated_images):
        """Calculate gradient penalty"""
        batch_size = real_images.shape[0]
        epsilon = tf.random.uniform([batch_size, 1, 1, 1], 0.0, 1.0)
        interpolated_images = epsilon * real_images + (1 - epsilon) * generated_images
        with tf.GradientTape() as tape:
            tape.watch(interpolated_images)
            interpolated_logits = self.discriminator(interpolated_images, training=True)
        grads = tape.gradient(interpolated_logits, interpolated_images)
        grad_norms = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=[1, 2, 3]))
        return tf.reduce_mean((grad_norms - 1.0) ** 2)

    def train_step(self, images, batch_size):
        # Sample latent vector
        latent_vector = tf.random.normal([batch_size, self.latent_vector_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate images with latent vector
            generated_images = self.generator(latent_vector, training=True)

            # Feed generated images and real training images through discriminator
            real_disc_output = self.discriminator(images, training=True)
            fake_disc_output = self.discriminator(generated_images, training=True)

            # Calculate discriminator loss and apply gradient penalty
            disc_loss = discriminator_loss(real_disc_output, fake_disc_output)
            gp = self.gradient_penalty(images, generated_images)
            disc_loss += self.lambda_gp * gp

            # Calculate generator loss
            gen_loss = generator_loss(fake_disc_output)

            # Calculate generator and discriminator gradients
            generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            # Update weights using gradients
            self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(zip(discriminator_gradients, self.discriminator.trainable_variables))

        return gen_loss.numpy(), disc_loss.numpy(), 0  # 0 for the placeholder accuracy since it doesn't apply to WGAN_GP
