import os
import random
from PIL import Image, ImageDraw, ImageFont

# Define the three classes and their colors (R, G, B)
CLASSES = {
    "Urban": (255, 0, 0),   # Red
    "Forest": (0, 153, 0),  # Green
    "Water": (0, 0, 255),   # Blue
}

def create_color_mask(width, height, classes, num_shapes_per_class=20, shape_size_range=(10, 50)):
    """
    Creates a random, semi-transparent color-coded mask with the provided classes.
    Each class will have randomly drawn rectangles on the mask.
    
    Parameters:
      - width, height: Dimensions of the mask (same as original image).
      - classes: Dictionary mapping class names to RGB tuples.
      - num_shapes_per_class: How many shapes to draw for each class.
      - shape_size_range: Tuple of (min_size, max_size) for a shape.
      
    Returns:
      A PIL Image (RGBA) with the colored shapes.
    """
    mask = Image.new("RGBA", (width, height), color=(0, 0, 0, 0))
    draw = ImageDraw.Draw(mask, "RGBA")
    
    # For each class, draw a number of random rectangles.
    for cls_name, cls_color in classes.items():
        for _ in range(num_shapes_per_class):
            shape_size = random.randint(shape_size_range[0], shape_size_range[1])
            x1 = random.randint(0, width - shape_size)
            y1 = random.randint(0, height - shape_size)
            x2 = x1 + shape_size
            y2 = y1 + shape_size
            
            # Use an alpha value to set the transparency of the overlay (0-255)
            draw.rectangle([x1, y1, x2, y2], fill=(cls_color[0], cls_color[1], cls_color[2], 180))
    
    return mask

def overlay_color_mask(base_image, color_mask):
    """
    Overlays the color mask onto the base image.
    
    Parameters:
      - base_image: The original image (PIL Image).
      - color_mask: The generated RGBA mask.
      
    Returns:
      A new blended image with the overlay (converted to RGBA).
    """
    if base_image.mode != "RGBA":
        base_image = base_image.convert("RGBA")
    
    blended = Image.alpha_composite(base_image, color_mask)
    return blended

def draw_legend(image, classes, legend_pos=(10, 10), box_size=20, spacing=5):
    """
    Draws a simple legend on the image for the colored classes.
    
    Parameters:
      - image: The PIL image on which to draw.
      - classes: Dictionary mapping class names to RGB tuples.
      - legend_pos: Tuple (x, y) for legend starting position.
      - box_size: Size of the color swatch.
      - spacing: Space between legend entries.
    """
    draw = ImageDraw.Draw(image)
    
    # Use the default font; if you need a custom font, specify a TTF file here
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    x, y = legend_pos
    for cls_name, cls_color in classes.items():
        # Draw a filled rectangle for the color swatch
        draw.rectangle([x, y, x + box_size, y + box_size],
                       fill=(cls_color[0], cls_color[1], cls_color[2], 255))
        # Write the class label to the right of the swatch
        draw.text((x + box_size + spacing, y), cls_name, fill=(0, 0, 0), font=font)
        
        y += box_size + spacing

def generate_fake_urban_images_color(input_dir, output_dir, classes,
                                     num_shapes_per_class=20, shape_size_range=(10, 50)):
    """
    Processes all images in input_dir, creates a synthetic color-coded overlay
    (using the provided classes), draws a legend, and saves the output image 
    with the same filename into output_dir.
    
    Parameters:
      - input_dir: Directory with original images.
      - output_dir: Directory to save processed images.
      - classes: Dictionary mapping class names to RGB values.
      - num_shapes_per_class, shape_size_range: Parameters for generating mask shapes.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    for file_name in os.listdir(input_dir):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp')):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)
            
            with Image.open(input_path) as base_img:
                width, height = base_img.size
                
                # Create a random color-coded mask for the classes
                color_mask = create_color_mask(width, height, classes,
                                               num_shapes_per_class=num_shapes_per_class,
                                               shape_size_range=shape_size_range)
                
                # Overlay the mask on top of the base image
                blended = overlay_color_mask(base_img, color_mask)
                
                # Draw the legend in the top-left corner
                draw_legend(blended, classes, legend_pos=(10, 10))
                
                # Convert to RGB (optional, if you wish to remove the alpha channel)
                final_img = blended.convert("RGB")
                final_img.save(output_path)
                print(f"Saved color-coded image to {output_path}")

def main():
    # Define input folders for preflood and postflood images
    preflood_dir = r"C:\Users\aayus\OneDrive\Desktop\s_dataset\preflood_dataset"
    postflood_dir = r"C:\Users\aayus\OneDrive\Desktop\s_dataset\postflood_dataset"

    # Define output folders
    output_preflood_dir = r"C:\Users\aayus\OneDrive\Desktop\s_dataset\fake_urban_color_dataset\preflood_urban_color"
    output_postflood_dir = r"C:\Users\aayus\OneDrive\Desktop\s_dataset\fake_urban_color_dataset\postflood_urban_color"
    
    # Generate color-coded images for the preflood dataset
    print("Generating color-coded images for pre-flood dataset...")
    generate_fake_urban_images_color(
        input_dir=preflood_dir,
        output_dir=output_preflood_dir,
        classes=CLASSES,
        num_shapes_per_class=20,
        shape_size_range=(10, 50)
    )
    
    # Generate color-coded images for the postflood dataset
    print("Generating color-coded images for post-flood dataset...")
    generate_fake_urban_images_color(
        input_dir=postflood_dir,
        output_dir=output_postflood_dir,
        classes=CLASSES,
        num_shapes_per_class=30,  # example: more shapes for post flood imagery
        shape_size_range=(20, 60)
    )

if __name__ == "__main__":
    main()
