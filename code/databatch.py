import os

base_dir = '/Users/danielpark/Projects/Vitis_AI_Model/Vitis-AI/model_zoo/pt_resnet50_0.5_3.5/data/Imagenet/val'
class_names = ['class1', 'class2']

# Create base directory if it doesn't exist
os.makedirs(base_dir, exist_ok=True)

# Create class directories and add dummy images
for class_name in class_names:
    class_dir = os.path.join(base_dir, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # Create dummy images
    for i in range(5):  # Create 5 dummy images per class
        with open(os.path.join(class_dir, f'img{i}.jpg'), 'w') as f:
            f.write('dummy image content')

print("Sample val directory structure created.")