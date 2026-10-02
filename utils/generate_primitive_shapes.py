import os
import random
from PIL import Image, ImageDraw

def generate_shape_dataset(base_dir="dataset/shape_dataset",samples_per_class=1000, img_size=64,):
    classes = ['circles', 'squares']
    
    # Create directory structure
    for cls in classes:
        os.makedirs(os.path.join(base_dir, cls), exist_ok=True)
        
    for i in range(samples_per_class):
        # 1. Randomize parameters to prevent model overfitting
        shape_size = random.randint(15, img_size - 10)
        x0 = random.randint(0, img_size - shape_size)
        y0 = random.randint(0, img_size - shape_size)
        x1, y1 = x0 + shape_size, y0 + shape_size
        
        # Random RGB color
        color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        
        # 2. Generate Circle
        img_circle = Image.new('RGB', (img_size, img_size), 'white')
        draw_circle = ImageDraw.Draw(img_circle)
        draw_circle.ellipse([x0, y0, x1, y1], fill=color)
        img_circle.save(os.path.join(base_dir, 'circles', f'circle_{i}.jpg'))
        
        # 3. Generate Square
        img_square = Image.new('RGB', (img_size, img_size), 'white')
        draw_square = ImageDraw.Draw(img_square)
        draw_square.rectangle([x0, y0, x1, y1], fill=color)
        img_square.save(os.path.join(base_dir, 'squares', f'square_{i}.jpg'))

    print(f"Dataset generated! {samples_per_class * 2} images saved to '{base_dir}/'")

if __name__ == "__main__":
    generate_shape_dataset()