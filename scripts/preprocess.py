import os
import json
from PIL import Image

def load_annotations(annotation_file):
    """Loads the annotation file and returns a dictionary mapping image IDs to captions."""
    with open(annotation_file, 'r') as f:
        captions = json.load(f)
    
    image_id_to_captions = {}
    for annotation in captions['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        if image_id in image_id_to_captions:
            image_id_to_captions[image_id].append(caption)
        else:
            image_id_to_captions[image_id] = [caption]
    
    return image_id_to_captions

def load_image(image_id, split='train'):
    """Loads an image given its image ID and split (train/val)."""
    image_folder = f'./data/{split}2014/'
    image_file = f'COCO_train2014_000000{image_id:06d}.jpg'
    image_path = os.path.join(image_folder, image_file)
    
    return Image.open(image_path)

# Example of loading an image
image_id = 458752  # Replace with any valid image ID
image = load_image(image_id)
image.show()
