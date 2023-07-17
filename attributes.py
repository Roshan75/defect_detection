import os
import numpy as np
import glob

def image_path_generator(img_dir):
    classes = os.listdir(img_dir)
    # Get the image file paths in the class folder
    for cls in classes:
        image_paths = glob.glob(f"{img_dir}/{cls}/*.png")  # Adjust the file extension if needed

        # Save the image paths as images.npy
        np.save(f"{img_dir}/{cls}/images.npy", image_paths)
    print(f'Image path file saved at classes of dataset: {img_dir}')

def attribute_generate():
    class_attributes = {
        0: [1, 0, 0, 0, 0, 1],
        1: [0, 1, 0, 0, 0, 1],
        2: [0, 0, 0, 0, 0, 1],
        3: [0, 0, 1, 0, 0, 1],
        4: [0, 0, 0, 1, 0, 1],
        5: [1, 0, 0, 0, 1, 1],
    }

    # Convert attributes dictionary to attribute_vectors array
    num_classes = len(class_attributes)
    vector_length = len(next(iter(class_attributes.values())))
    attribute_vectors = np.zeros((num_classes, vector_length))

    for class_id, attributes in class_attributes.items():
        attribute_vectors[class_id] = attributes

    # Save attribute_vectors as attribute_vectors.npy
    np.save("../models/attribute_vectors.npy", attribute_vectors)

if __name__ == "__main__":
    #image_path_generator(r"E:\thoucentric\defect_detection\dataset\carpet\test")
    attribute_generate()