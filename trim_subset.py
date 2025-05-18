import os
import glob

# Set this to your subset path
subset_path = "data/Subset"
max_images = 100  # Number of images to keep per class

for class_folder in os.listdir(subset_path):
    class_path = os.path.join(subset_path, class_folder)
    if os.path.isdir(class_path):
        images = glob.glob(os.path.join(class_path, "*"))
        images.sort()  # Sort to keep consistent order

        if len(images) > max_images:
            print(f"Trimming {class_folder}: keeping first {max_images} of {len(images)} images.")
            for img_path in images[max_images:]:
                try:
                    os.remove(img_path)
                except Exception as e:
                    print(f"Failed to delete {img_path}: {e}")
        else:
            print(f"{class_folder}: only {len(images)} images, no trimming needed.")
