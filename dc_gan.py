import tensorflow as tf
from gan import GAN

# Define cross entropy function to be used in loss function. from_logits=True so logits expected from discriminator
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(fake_output):
    """DC GAN Generator Loss Function"""
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def discriminator_loss(real_output, fake_output):
    """DC GAN Discriminator Loss Function"""
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def discriminator_accuracy(real_output, fake_output):
    """Method to calculate discriminator accuracy"""
    # Convert logit discriminator outputs to either 0 or 1, and combine into predicted_labels array
    real_predicted_labels = tf.round(tf.sigmoid(real_output))
    fake_predicted_labels = tf.round(tf.sigmoid(fake_output))
    predicted_labels = tf.concat([real_predicted_labels, fake_predicted_labels], axis=0)

    # Create real labels for data. 1 for real and 0 for fake
    real_labels = tf.ones_like(real_output)
    fake_labels = tf.zeros_like(fake_output)
    true_labels = tf.concat([real_labels, fake_labels], axis=0)

    # Calculate and return accuracy
    correct_predictions = tf.equal(predicted_labels, true_labels)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy.numpy()


class DC_GAN(GAN):
    def __init__(self, generator, discriminator, latent_vector_dim, folder_path,
                 gen_learning_rate=1e-4, disc_learning_rate=1e-4):
        super().__init__(generator, discriminator, latent_vector_dim, folder_path,
                         gen_learning_rate, disc_learning_rate)

    def train_step(self, images, batch_size):
        # Sample latent vector
        latent_vector = tf.random.normal([batch_size, self.latent_vector_dim])

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate images with latent vector
            generated_images = self.generator(latent_vector, training=True)

            # Feed generated images and real training images through discriminator
            real_disc_output = self.discriminator(images, training=True)
            fake_disc_output = self.discriminator(generated_images, training=True)

            # Calculate losses and accuracy
            gen_loss = generator_loss(fake_disc_output)
            disc_loss = discriminator_loss(real_disc_output, fake_disc_output)
            disc_accuracy = discriminator_accuracy(real_disc_output, fake_disc_output)

            # Calculate generator and discriminator gradients
            generator_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
            discriminator_gradients = disc_tape.gradient(disc_loss, self.discriminator.trainable_variables)

            # Update weights using gradients
            self.generator_optimizer.apply_gradients(zip(generator_gradients, self.generator.trainable_variables))
            self.discriminator_optimizer.apply_gradients(
                zip(discriminator_gradients, self.discriminator.trainable_variables))

        # Return losses and accuracy
        return gen_loss.numpy(), disc_loss.numpy(), disc_accuracy
