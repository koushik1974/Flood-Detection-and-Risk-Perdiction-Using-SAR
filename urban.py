import os
import random
from PIL import Image, ImageDraw, ImageFont

# Define a dictionary of classes and their corresponding colors (R, G, B)
LCZ_COLORS = {
    "Urban": (255, 0, 0),      # Red
    "Forest": (0, 153, 0),     # Green
    "Water": (0, 0, 255),      # Blue
    "Barren": (153, 102, 51),  # Brownish
    "Other": (255, 255, 0),    # Yellow
}

def create_color_mask(width, height, classes, num_shapes_per_class=20, shape_size_range=(10, 50)):
    """
    Creates a random color-coded mask of different classes for demonstration.
    Each 'class' will appear as a set of random colored shapes in the image.
    """
    # Start with a blank RGBA image
    mask = Image.new("RGBA", (width, height), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(mask, "RGBA")
    
    for cls_name, cls_color in classes.items():
        for _ in range(num_shapes_per_class):
            shape_size = random.randint(shape_size_range[0], shape_size_range[1])
            
            x1 = random.randint(0, width - shape_size)
            y1 = random.randint(0, height - shape_size)
            x2 = x1 + shape_size
            y2 = y1 + shape_size
            
            # Draw rectangle with the class color
            # We set alpha=180 (semi-transparent); adjust as needed
            draw.rectangle([x1, y1, x2, y2], fill=(cls_color[0], cls_color[1], cls_color[2], 180))
    
    return mask

def overlay_color_mask(base_image, color_mask):
    """
    Overlays a color mask onto the base image (which may be grayscale).
    Returns a new RGBA image with the overlay applied.
    """
    # Convert base image to RGBA if not already
    if base_image.mode != "RGBA":
        base_image = base_image.convert("RGBA")

    # Alpha composite the base_image with the mask
    # This effectively places the colored shapes on top of the original
    blended = Image.alpha_composite(base_image, color_mask)
    return blended

def draw_legend(image, classes, legend_pos=(10, 10), box_size=20, spacing=5, font_size=16):
    """
    Draws a simple legend on the image for the color-coded classes.
    - legend_pos: (x, y) starting point for the legend box
    - box_size: size of the color swatch
    - spacing: spacing between legend rows
    - font_size: size of text (requires Pillow's default or custom font)
    """
    draw = ImageDraw.Draw(image)
    
    # Attempt to load a default system font
    try:
        font = ImageFont.load_default()
    except:
        font = None

    x, y = legend_pos
    for cls_name, cls_color in classes.items():
        # Draw a small color box
        draw.rectangle([x, y, x+box_size, y+box_size],
                       fill=(cls_color[0], cls_color[1], cls_color[2], 255))
        
        # Put class name next to the box
        text_x = x + box_size + spacing
        text_y = y
        draw.text((text_x, text_y), cls_name, fill=(0, 0, 0), font=font)
        
        # Move downward for next label
        y += box_size + spacing

def generate_fake_urban_images_color(input_dir, output_dir,
                                     classes_dict,
                                     num_shapes_per_class=20,
                                     shape_size_range=(10, 50)):
    """
    Reads images from `input_dir`, generates a color-coded overlay, and
    saves them to `output_dir` with the same file names.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            
            # Open base image
            with Image.open(input_path) as base_img:
                width, height = base_img.size
                
                # Create a random color-coded mask
                color_mask = create_color_mask(width, height,
                                               classes_dict,
                                               num_shapes_per_class=num_shapes_per_class,
                                               shape_size_range=shape_size_range)
                
                # Overlay it
                blended = overlay_color_mask(base_img, color_mask)
                
                # Draw legend onto the blended image
                draw_legend(blended, classes_dict, legend_pos=(10, 10))
                
                # Save the final image (RGBA or convert to PNG)
                # If you want to keep alpha channel, keep "RGBA", else convert to "RGB"
                blended_rgb = blended.convert("RGB")
                blended_rgb.save(output_path)
                
                print(f"Saved color-coded urban image to {output_path}")

def main():
    preflood_dir = r"C:\Users\aayus\OneDrive\Desktop\s_dataset\preflood_dataset"
    postflood_dir = r"C:\Users\aayus\OneDrive\Desktop\s_dataset\postflood_dataset"

    # Output directories
    fake_urban_color_preflood_dir = r"C:\Users\aayus\OneDrive\Desktop\s_dataset\fake_urban_color_dataset\preflood_urban_color"
    fake_urban_color_postflood_dir = r"C:\Users\aayus\OneDrive\Desktop\s_dataset\fake_urban_color_dataset\postflood_urban_color"

    # Our color classes (adjust the dictionary as needed)
    classes_dict = {
        "Urban": (255, 0, 0),
        "Forest": (0, 153, 0),
        "Water": (0, 0, 255),
        "Barren": (153, 102, 51),
        "Other": (255, 255, 0),
    }

    # Generate color-coded fake images for preflood
    print("Generating color-coded images for pre-flood dataset...")
    generate_fake_urban_images_color(
        input_dir=preflood_dir,
        output_dir=fake_urban_color_preflood_dir,
        classes_dict=classes_dict,
        num_shapes_per_class=20,
        shape_size_range=(10, 50)
    )
    
    # Generate color-coded fake images for postflood
    print("Generating color-coded images for post-flood dataset...")
    generate_fake_urban_images_color(
        input_dir=postflood_dir,
        output_dir=fake_urban_color_postflood_dir,
        classes_dict=classes_dict,
        num_shapes_per_class=30,
        shape_size_range=(20, 60)
    )

if __name__ == "__main__":
    main()
