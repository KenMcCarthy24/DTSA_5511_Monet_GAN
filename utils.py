import numpy as np
import matplotlib.pyplot as plt


def display_random_images(image_data_array, N):
    """Displays N random images from the given image_data_array"""
    # Calculate the grid size
    grid_size = int(np.ceil(np.sqrt(N)))

    # Randomly select N images
    selected_images = np.random.choice(image_data_array.shape[0], N, replace=False)

    fig, axs = plt.subplots(grid_size, grid_size, figsize=(7, 7))

    for i, ax in enumerate(axs.flatten()):
        if i < N:
            ax.imshow(image_data_array[selected_images[i]])
            ax.axis('off')  # Turn off axis for each subplot
        else:
            fig.delaxes(ax)  # Remove excess subplots

    plt.tight_layout()
    plt.show()