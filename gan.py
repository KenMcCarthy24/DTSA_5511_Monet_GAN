import tensorflow as tf
import os
import io
import matplotlib.pyplot as plt
import numpy as np

from matplotlib import gridspec
from PIL import Image
from zipfile import ZipFile

from utils import display_random_images

import json
import shutil
import time


class GAN:
    def __init__(self, generator, discriminator, latent_vector_dim, folder_path,
                 gen_learning_rate=1e-4, disc_learning_rate=1e-4):
        self.generator = generator
        self.discriminator = discriminator
        self.latent_vector_dim = latent_vector_dim
        self.folder_path = folder_path

        self.generator_optimizer = tf.keras.optimizers.Adam(gen_learning_rate)
        self.discriminator_optimizer = tf.keras.optimizers.Adam(disc_learning_rate)

        self.history = dict(gen_loss=[], disc_loss=[], disc_accuracy=[], time=[])

        self.checkpoint = tf.train.Checkpoint(generator=generator,
                                              discriminator=discriminator,
                                              generator_optimizer=self.generator_optimizer,
                                              discriminator_optimizer=self.discriminator_optimizer)

    def train_step(self, images, batch_size):
        """Method for doing a single step of training, meant to be overwritten by different GAN implementations"""
        pass

    def fit(self, X_train, batch_size, epochs, load_from_file, load_epoch=None):
        """Trains GAN on training data with the given batch size and number of epochs"""
        print("Starting Training")

        # Define model and history path
        checkpoint_folder = os.path.join(self.folder_path, "train_checkpoints")
        history_path = os.path.join(self.folder_path, "train_history.json")

        # If load_from_file = True, don't train new GAN just load previously saved weights and history from file
        if load_from_file:
            print("Loading Model From File")

            if load_epoch:
                # Load weights from a specific epoch
                checkpoint_path = os.path.join(checkpoint_folder, f"Epoch{load_epoch}-{int(load_epoch/100)}")
                self.checkpoint.restore(checkpoint_path)
            else:
                # Default behavior: load the latest checkpoint
                self.checkpoint.restore(tf.train.latest_checkpoint(checkpoint_folder))

            with open(history_path) as f:
                self.history = json.load(f)

        else:
            # Prepare output folder structure
            os.makedirs(self.folder_path, exist_ok=True)
            shutil.rmtree(self.folder_path)
            train_images_folder = os.path.join(self.folder_path, "train_images")
            os.makedirs(train_images_folder)

            # Run each epoch
            for epoch in range(epochs):
                epoch_num = epoch + 1

                start = time.time()

                # Initialize loss and accuracy metrics for this epoch
                gen_loss, disc_loss, disc_accuracy = 0, 0, 0

                # Separate data into batches and run a train step on each, keeping track of the metrics
                N_batches = int(X_train.shape[0] / batch_size)
                for n in range(N_batches):
                    batch = X_train[n * batch_size:(n + 1) * batch_size]
                    gen_loss_batch, disc_loss_batch, disc_accuracy_batch = self.train_step(batch, batch_size)
                    gen_loss += gen_loss_batch
                    disc_loss += disc_loss_batch
                    disc_accuracy += disc_accuracy_batch

                # Average all metrics over the N batches and add to history
                gen_loss = gen_loss / N_batches
                disc_loss = disc_loss / N_batches
                disc_accuracy = disc_accuracy / N_batches
                end = time.time()
                self.history["gen_loss"].append(gen_loss)
                self.history["disc_loss"].append(disc_loss)
                self.history["disc_accuracy"].append(disc_accuracy)
                self.history["time"].append(end - start)

                # Every 100 epochs, save a checkpoint of model progress and generate an image to track progress, also save history
                test_latent_vector = tf.random.normal([1, self.latent_vector_dim])
                if epoch_num % 100 == 0:
                    print(f"Epoch {epoch_num} finished")
                    self.checkpoint.save(file_prefix=os.path.join(checkpoint_folder, f"Epoch{epoch_num}"))
                    self.generate_image_and_save_to_file(test_latent_vector,
                                                         os.path.join(train_images_folder, f"Epoch{epoch_num}.png"))
                    with open(history_path, "w+") as f:
                        json.dump(self.history, f)

            # Save final history after training
            with open(history_path, "w+") as f:
                json.dump(self.history, f)

    def generate_image(self, latent_vector):
        """Use this GAN's generator to generate a single image using provided latent vector, also un-normalize it"""
        image_array = (self.generator(latent_vector)[0]).numpy()

        return np.round(((image_array + 1) / 2) * 255).astype(np.uint8)

    def generate_image_and_save_to_file(self, latent_vector, out_file):
        """Use provided latent vector to generate an image and save it to provided file"""
        image_array = self.generate_image(latent_vector)
        image = Image.fromarray(image_array)
        image.save(out_file)

    def make_submission(self, output_folder_path):
        """Generate 10000 images from the generator in correct format for Kaggle competition"""
        # Ensure the output folder exists
        os.makedirs(output_folder_path, exist_ok=True)

        # Create a zip file in the specified output folder
        zip_file_path = os.path.join(output_folder_path, "images.zip")
        with ZipFile(zip_file_path, 'w') as zipf:
            n_files = 10000
            for i in range(n_files):
                # Generate image
                image_array = self.generate_image(tf.random.normal([1, self.latent_vector_dim]))

                # Convert the NumPy array to PIL Image
                image = Image.fromarray(image_array)

                # Save the image to an in-memory BytesIO object
                img_data = io.BytesIO()
                image.save(img_data, format='JPEG')  # Change format to JPEG

                # Write the image data to the zip file with a filename
                img_name = f"{i}.jpg"  # Change file extension to .jpg
                zipf.writestr(img_name, img_data.getvalue())

    def plot_history(self):
        """Makes a plot of generator loss, discriminator loss, and discriminator accuracy across the training epochs"""
        epochs = list(range(1, len(self.history["gen_loss"]) + 1))

        fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6))

        # Plot Generator Loss
        axes[0].plot(epochs, self.history["gen_loss"])
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Generator Loss')

        # Plot Discriminator Loss
        axes[1].plot(epochs, self.history["disc_loss"])
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('Discriminator Loss')

        # Plot Discriminator Accuracy
        axes[2].plot(epochs, self.history["disc_accuracy"])
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('Accuracy')
        axes[2].set_title('Discriminator Accuracy')

        # Display the plots
        plt.tight_layout()
        plt.show()

    def visualize_training_images(self):
        """Visualize the images that were output every 100 epochs during training"""
        train_images_folder = os.path.join(self.folder_path, "train_images")

        image_files = [f for f in os.listdir(train_images_folder) if f.endswith('.png')]

        image_files.sort(key=lambda f: int(f.split('Epoch')[1].split('.')[0]))

        # Determine number of rows, always use five columns
        n = len(image_files)
        cols = 5
        rows = (n // cols) + (1 if n % cols else 0)

        # Create figure
        fig = plt.figure(figsize=(15, rows * 3))
        gs = gridspec.GridSpec(rows, cols)
        for i, image_file in enumerate(image_files):
            image_path = os.path.join(train_images_folder, image_file)
            image = Image.open(image_path)

            epoch_num = int(image_file.split('Epoch')[1].split('.')[0])

            ax = plt.subplot(gs[i])
            ax.imshow(image)
            ax.set_title(f'Epoch {epoch_num}')
            ax.axis('off')

        plt.show()

    def visualize_random_images(self):
        """Generates 25 random images using the generator and displays them on a grid"""
        arr = self.generator(tf.random.normal([25, self.latent_vector_dim])).numpy()
        display_random_images(np.round(((arr + 1) / 2) * 255).astype(np.uint8), 25)