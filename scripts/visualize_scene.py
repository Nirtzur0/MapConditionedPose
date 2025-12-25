
import sionna
import os
from sionna.rt import load_scene, Camera
import matplotlib.pyplot as plt

def visualize_scene(scene_path, output_path):
    """
    Visualizes a Sionna scene and saves the plot.

    Args:
        scene_path (str): Path to the scene.xml file.
        output_path (str): Path to save the visualization.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Load the scene
    scene = load_scene(scene_path)

    # Create a camera
    camera = Camera("my_camera", position=[0, 0, 100], look_at=[0, 0, 0])
    scene.add(camera)
    scene.camera = "my_camera"

    # Plot the scene
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    scene.plot(ax=ax)
    plt.savefig(output_path)
    plt.close(fig)

if __name__ == "__main__":
    scene_to_visualize = "data/scenes/austin_texas/scene_-97.765_30.27/scene.xml"
    output_image_path = "visualizations/austin_scene.png"
    
    visualize_scene(scene_to_visualize, output_image_path)
    print(f"Scene visualization saved to {output_image_path}")
