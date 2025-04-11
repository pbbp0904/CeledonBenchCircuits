import os
import random
import math
import uuid
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# --- Constants ---
IMG_SIZE = 1024
BACKGROUND_COLOR = (255, 255, 255)
SHAPE_TYPES = ["circle", "square", "triangle", "rectangle", "pentagon", "star"]
COLORS = {
    "red": (255, 0, 0),
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "purple": (128, 0, 128),
    "orange": (255, 165, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "lime": (0, 128, 0),
    "pink": (255, 192, 203),
    "teal": (0, 128, 128),
    "brown": (165, 42, 42),
    "navy": (0, 0, 128),
    "maroon": (128, 0, 0),
    "olive": (128, 128, 0),
}
MIN_SHAPE_SIZE = 40
MAX_SHAPE_SIZE = 80
MIN_DISTANCE = 50 # Min distance between centers of any two objects (shapes or numbers)
BORDER_MARGIN = 50
FONT_SIZE = 30
ARROW_COLOR = (50, 50, 50)
ARROW_WIDTH = 3

# --- Helper Functions ---

def get_random_color_name():
    return random.choice(list(COLORS.keys()))

def get_random_shape_type():
    return random.choice(SHAPE_TYPES)

def is_overlapping(pos1, size1, pos2, size2, min_dist):
    """Check if two square bounding boxes overlap with a minimum distance buffer."""
    dist = math.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    # Check distance between centers against sum of half sizes + min_dist
    return dist < (size1 / 2 + size2 / 2 + min_dist)

def get_random_position(existing_objects, obj_size):
    """Find a random position that doesn't overlap with existing objects."""
    max_attempts = 200
    for _ in range(max_attempts):
        x = random.randint(BORDER_MARGIN + obj_size // 2, IMG_SIZE - BORDER_MARGIN - obj_size // 2)
        y = random.randint(BORDER_MARGIN + obj_size // 2, IMG_SIZE - BORDER_MARGIN - obj_size // 2)
        new_pos = (x, y)
        
        overlap = False
        for obj_pos, obj_sz in existing_objects:
            if is_overlapping(new_pos, obj_size, obj_pos, obj_sz, MIN_DISTANCE):
                overlap = True
                break
        
        if not overlap:
            return new_pos
            
    print(f"Warning: Could not find non-overlapping position after {max_attempts} attempts.")
    # Return a potentially overlapping position as fallback
    return (random.randint(BORDER_MARGIN + obj_size // 2, IMG_SIZE - BORDER_MARGIN - obj_size // 2),
            random.randint(BORDER_MARGIN + obj_size // 2, IMG_SIZE - BORDER_MARGIN - obj_size // 2))

def draw_shape(draw, shape_type, center_pos, size, color):
    """Draws a specified shape."""
    x, y = center_pos
    r = size // 2 # Use radius/half-size for calculations
    bbox = (x - r, y - r, x + r, y + r)

    if shape_type == "circle":
        draw.ellipse(bbox, fill=color)
    elif shape_type == "square":
        draw.rectangle(bbox, fill=color)
    elif shape_type == "rectangle":
        # Make rectangle distinct from square
        width_r = r
        height_r = int(r * (random.uniform(0.5, 0.8) if random.random() > 0.5 else random.uniform(1.2, 1.5)))
        if random.random() > 0.5: # Randomly swap width/height
            width_r, height_r = height_r, width_r
        rect_bbox = (x - width_r, y - height_r, x + width_r, y + height_r)
        draw.rectangle(rect_bbox, fill=color)
        bbox = rect_bbox # Update bbox for ground truth
    elif shape_type == "triangle": # Equilateral pointing up
        height = int(r * math.sqrt(3))
        p1 = (x, y - (2/3) * height / 2) # Adjusted for center
        p2 = (x - r, y + (1/3) * height / 2)
        p3 = (x + r, y + (1/3) * height / 2)
        draw.polygon([p1, p2, p3], fill=color)
        # Bbox is approximate for triangle
        bbox = (x - r, y - int((2/3) * height / 2), x + r, y + int((1/3) * height / 2))
    elif shape_type == "pentagon":
        points = []
        for i in range(5):
            angle = math.pi / 2 - 2 * math.pi * i / 5
            px = x + r * math.cos(angle)
            py = y - r * math.sin(angle) # PIL y increases downwards
            points.append((px, py))
        draw.polygon(points, fill=color)
    elif shape_type == "star": # Simple 5-pointed star
        points = []
        outer_r = r
        inner_r = r * 0.5
        for i in range(10):
            current_r = outer_r if i % 2 == 0 else inner_r
            angle = math.pi / 2 - 2 * math.pi * i / 10
            px = x + current_r * math.cos(angle)
            py = y - current_r * math.sin(angle)
            points.append((px, py))
        draw.polygon(points, fill=color)
        
    # Return actual bounding box used for drawing
    return bbox

def draw_arrow(draw, start_pos, end_pos, color, width):
    """Draws a highly complex, curvy (quartic Bezier) arrow from start to end."""
    x0, y0 = start_pos # P0
    x4, y4 = end_pos   # P4

    dx = x4 - x0
    dy = y4 - y0
    dist = math.sqrt(dx*dx + dy*dy)
    if dist == 0: return # Avoid division by zero

    # --- Calculate THREE control points for Quartic Bezier --- 
    
    # Base points along the line (e.g., 1/4, 1/2, 3/4)
    base_ctrl1_x = x0 + dx * 0.25
    base_ctrl1_y = y0 + dy * 0.25
    base_ctrl2_x = x0 + dx * 0.50
    base_ctrl2_y = y0 + dy * 0.50
    base_ctrl3_x = x0 + dx * 0.75
    base_ctrl3_y = y0 + dy * 0.75

    # Perpendicular vector
    perp_dx = -dy / dist
    perp_dy = dx / dist

    # Random offsets for control points
    max_offset_factor = 0.25 # Adjust max offset if needed
    offset1_mag = dist * random.uniform(0.05, max_offset_factor) * random.choice([-1, 1])
    offset2_mag = dist * random.uniform(0.05, max_offset_factor) * random.choice([-1, 1])
    offset3_mag = dist * random.uniform(0.05, max_offset_factor) * random.choice([-1, 1])
    
    # Calculate final control points (P1, P2, P3)
    ctrl1_x = base_ctrl1_x + offset1_mag * perp_dx
    ctrl1_y = base_ctrl1_y + offset1_mag * perp_dy
    ctrl2_x = base_ctrl2_x + offset2_mag * perp_dx
    ctrl2_y = base_ctrl2_y + offset2_mag * perp_dy
    ctrl3_x = base_ctrl3_x + offset3_mag * perp_dx
    ctrl3_y = base_ctrl3_y + offset3_mag * perp_dy
    
    ctrl1 = (ctrl1_x, ctrl1_y) # P1
    ctrl2 = (ctrl2_x, ctrl2_y) # P2
    ctrl3 = (ctrl3_x, ctrl3_y) # P3

    # --- Draw curve using multiple line segments --- 
    points = [start_pos]
    steps = 25 # More steps might be needed for smoother quartic curve
    for i in range(1, steps + 1):
        t = i / steps
        inv_t = 1 - t
        # Quartic Bezier formula: B(t) = P0*(1-t)^4 + P1*4*(1-t)^3*t + P2*6*(1-t)^2*t^2 + P3*4*(1-t)*t^3 + P4*t^4
        x = (inv_t**4 * x0) + \
            (4 * inv_t**3 * t * ctrl1_x) + \
            (6 * inv_t**2 * t**2 * ctrl2_x) + \
            (4 * inv_t * t**3 * ctrl3_x) + \
            (t**4 * x4)
        y = (inv_t**4 * y0) + \
            (4 * inv_t**3 * t * ctrl1_y) + \
            (6 * inv_t**2 * t**2 * ctrl2_y) + \
            (4 * inv_t * t**3 * ctrl3_y) + \
            (t**4 * y4)
        points.append((x, y))
    
    draw.line(points, fill=color, width=width)

    # --- Draw arrowhead --- 
    arrow_len = 15
    # Angle based on the direction from the third control point (P3) to the end point (P4)
    angle = math.atan2(y4 - ctrl3_y, x4 - ctrl3_x) 
    angle1 = angle + math.pi * 5 / 6
    angle2 = angle - math.pi * 5 / 6
    x1_head = x4 + arrow_len * math.cos(angle1)
    y1_head = y4 + arrow_len * math.sin(angle1)
    x2_head = x4 + arrow_len * math.cos(angle2)
    y2_head = y4 + arrow_len * math.sin(angle2)
    draw.polygon([end_pos, (x1_head, y1_head), (x2_head, y2_head)], fill=color)

def get_bbox_intersection(p1, p2, bbox):
    """Calculates the intersection point of the line segment p1-p2 with a bbox."""
    x1, y1 = p1
    x2, y2 = p2
    bx_min, by_min, bx_max, by_max = bbox

    # Check intersection with each bbox edge
    # Using Liang-Barsky algorithm idea (simplified for axis-aligned box)
    dx = x2 - x1
    dy = y2 - y1
    
    t0, t1 = 0.0, 1.0 # Parameter range for the line segment p1 -> p2

    # Check intersections with vertical lines (x = bx_min, x = bx_max)
    if dx != 0:
        tx_min = (bx_min - x1) / dx
        tx_max = (bx_max - x1) / dx
        if tx_min > tx_max: tx_min, tx_max = tx_max, tx_min # Ensure tx_min <= tx_max
        
        t0 = max(t0, tx_min)
        t1 = min(t1, tx_max)

    # Check intersections with horizontal lines (y = by_min, y = by_max)
    if dy != 0:
        ty_min = (by_min - y1) / dy
        ty_max = (by_max - y1) / dy
        if ty_min > ty_max: ty_min, ty_max = ty_max, ty_min # Ensure ty_min <= ty_max

        t0 = max(t0, ty_min)
        t1 = min(t1, ty_max)
        
    # If intersection parameter range is valid (t0 < t1 and within [0,1])
    # We want the intersection closest to p1 that is *on* the bbox border
    # This corresponds to the smallest positive t value where the line enters/exits the box
    
    intersect_t = t0 # Use the first intersection point along the line from p1
    
    if intersect_t > 1.0 or intersect_t < 0.0: # Ensure intersection is within the segment p1-p2
       # Fallback: line segment doesn't intersect bbox (shouldn't happen if p2 is center of bbox)
       # Or if p1 is inside bbox. Return the point on bbox edge closest to p1?
       # For simplicity, let's return the center of the edge closest to p1
       center_x = (bx_min + bx_max) / 2
       center_y = (by_min + by_max) / 2
       dist_left = abs(x1 - bx_min)
       dist_right = abs(x1 - bx_max)
       dist_top = abs(y1 - by_min)
       dist_bottom = abs(y1 - by_max)
       min_dist = min(dist_left, dist_right, dist_top, dist_bottom)
       if min_dist == dist_left: return (bx_min, center_y)
       if min_dist == dist_right: return (bx_max, center_y)
       if min_dist == dist_top: return (center_x, by_min)
       return (center_x, by_max) # Must be bottom
       
    ix = x1 + intersect_t * dx
    iy = y1 + intersect_t * dy
    
    # Clamp to box boundary just in case of floating point issues
    ix = max(bx_min, min(bx_max, ix))
    iy = max(by_min, min(by_max, iy))
    
    return (ix, iy)

# --- Main Generator Function ---

def generate_circuit_image(N: int, image_save_path: str, seed: int | None = None):
    """Generates a circuit-like image with N shapes, numbers, and connecting arrows.

    Args:
        N: The number of shape/number pairs.
        image_save_path: The full path where the generated image should be saved.
        seed: An optional integer seed for the random number generator for deterministic output.

    Returns:
        A dictionary containing the ground truth data for the generated image, or None if generation fails.
    """
    if seed is not None:
        print(f"Generating image for N={N} using seed={seed}...")
        random.seed(seed)
    else:
        print(f"Generating image for N={N} (random seed)...")
        
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)
    
    # Attempt to load a default font. Handle potential errors.
    try:
        # Try a common generic font name first
        font = ImageFont.truetype("arial.ttf", FONT_SIZE)
    except IOError:
        try:
            # Try a fallback (often available on Linux/macOS)
            font = ImageFont.truetype("DejaVuSans.ttf", FONT_SIZE)
        except IOError:
            # If still not found, use PIL's default bitmap font
            print("Warning: TTF font not found. Using PIL default font.")
            font = ImageFont.load_default()

    ground_truth = {
        "n_value": N,
        "image_filename": os.path.basename(image_save_path),
        "shapes": [],
        "numbers": [],
        "connections": {}
    }
    
    existing_object_positions = [] # List of ((center_x, center_y), size) tuples

    # Determine subsets of shapes and colors based on N
    available_shapes = list(SHAPE_TYPES)
    available_color_names = list(COLORS.keys())

    n_shapes_to_use = min(N, len(available_shapes))
    n_colors_to_use = min(N, len(available_color_names))

    if n_shapes_to_use < N:
        print(f"Warning: N={N} is larger than the number of available shape types ({len(available_shapes)}). Using all {n_shapes_to_use} available shape types.")
    if n_colors_to_use < N:
         print(f"Warning: N={N} is larger than the number of available colors ({len(available_color_names)}). Using all {n_colors_to_use} available colors.")

    # Select the first N (or fewer if not enough) shapes and colors
    selected_shapes = available_shapes[:n_shapes_to_use]
    selected_color_names = available_color_names[:n_colors_to_use]
    
    # Ensure each of the N shapes and N colors appears exactly once if possible.
    # Shuffle the selected lists so the pairing is random.
    random.shuffle(selected_shapes)
    random.shuffle(selected_color_names)
    
    # If N > available shapes/colors, pad the lists by repeating elements (warning already printed).
    # This allows the loop below to run N times without index errors.
    while len(selected_shapes) < N:
        selected_shapes.append(random.choice(available_shapes[:n_shapes_to_use])) # Reuse from the selected pool
    while len(selected_color_names) < N:
        selected_color_names.append(random.choice(available_color_names[:n_colors_to_use])) # Reuse from the selected pool

    # 1. Place Shapes
    shapes_data = []
    for i in range(N):
        shape_id = f"shape_{uuid.uuid4()}"
        shape_size = random.randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
        
        # Assign shape type and color from the shuffled N-sized lists
        shape_type = selected_shapes[i]
        color_name = selected_color_names[i]
            
        color_rgb = COLORS[color_name]
        center_pos = get_random_position(existing_object_positions, shape_size)
        shape_bbox = draw_shape(draw, shape_type, center_pos, shape_size, color_rgb)
        
        shape_info = {
            "id": shape_id,
            "type": shape_type,
            "color": color_name,
            "size_parameter": shape_size, # Original target size
            "center_position": center_pos,
            "bounding_box": shape_bbox # Actual drawn bbox
        }
        shapes_data.append(shape_info)
        existing_object_positions.append((center_pos, max(shape_bbox[2]-shape_bbox[0], shape_bbox[3]-shape_bbox[1]))) # Use max dimension of bbox as size

    ground_truth["shapes"] = shapes_data

    # 2. Place Numbers
    numbers_data = []
    for i in range(N):
        number_val = i + 1
        text = str(number_val)
        # Replace textsize with textbbox
        # Calculate text dimensions using textbbox anchored at (0,0)
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        num_bbox_size = max(text_width, text_height) # Use max dimension for spacing
        
        center_pos = get_random_position(existing_object_positions, num_bbox_size)
        
        # Use anchor='mm' to draw text centered at center_pos
        # Ensure the Pillow version supports the anchor argument.
        try:
            draw.text(center_pos, text, fill=(0, 0, 0), font=font, anchor="mm")
            # If successful, draw_pos is effectively center_pos for anchoring purposes
            # but we still need the bbox dimensions for collision/connection
        except TypeError:
            # Fallback if anchor is not supported (older Pillow versions)
            print("Warning: Pillow version might not support text anchor. Falling back to manual centering.")
            draw_pos = (center_pos[0] - text_width // 2, center_pos[1] - text_height // 2)
            draw.text(draw_pos, text, fill=(0, 0, 0), font=font)
        
        # Calculate the bounding box based on the center position and font metrics
        # This relies on textbbox correctly representing the space needed around the center
        # Using floating point for potentially better accuracy before storing as tuple
        half_width = text_width / 2
        half_height = text_height / 2
        num_bbox = (
            center_pos[0] - half_width,
            center_pos[1] - half_height,
            center_pos[0] + half_width,
            center_pos[1] + half_height
        )
        
        number_info = {
            "value": number_val,
            "center_position": center_pos, # This is the intended visual center
            "bounding_box": num_bbox       # Bbox based on font metrics, centered around center_pos
        }
        numbers_data.append(number_info)
        existing_object_positions.append((center_pos, num_bbox_size))

    ground_truth["numbers"] = numbers_data

    # 3. Draw Arrows & Record Connections
    for i in range(N):
        number_info = numbers_data[i]
        shape_info = shapes_data[i]
        
        # Get centers and bounding boxes
        num_center = number_info["center_position"]
        shape_center = shape_info["center_position"]
        num_bbox = number_info["bounding_box"]
        shape_bbox = shape_info["bounding_box"]
        
        # Calculate arrow start point (intersection with number bbox)
        # Line goes from shape center towards number center
        start_arrow_pos = get_bbox_intersection(p1=shape_center, p2=num_center, bbox=num_bbox)
        
        # Calculate arrow end point (intersection with shape bbox)
        # Line goes from number center towards shape center
        end_arrow_pos = get_bbox_intersection(p1=num_center, p2=shape_center, bbox=shape_bbox)
        
        # Original center-to-center connection (commented out)
        # start_arrow_pos = number_info["center_position"]
        # end_arrow_pos = shape_info["center_position"]
        
        draw_arrow(draw, start_arrow_pos, end_arrow_pos, ARROW_COLOR, ARROW_WIDTH)
        
        ground_truth["connections"][number_info["value"]] = shape_info["id"]

    # 4. Save Image
    try:
        img.save(image_save_path)
        print(f"Successfully generated and saved image: {image_save_path}")
        return ground_truth
    except Exception as e:
        print(f"Error saving image {image_save_path}: {e}")
        return None

# --- Example Usage (for testing) ---
if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_output_dir = os.path.join(script_dir, "..", "..", "test_image_output")
    os.makedirs(test_output_dir, exist_ok=True)
    
    test_n = 5
    test_image_path = os.path.join(test_output_dir, f"test_circuit_n{test_n}.png")
    
    gt_data = generate_circuit_image(test_n, test_image_path)
    
    if gt_data:
        print("\nGround Truth Data:")
        import json
        print(json.dumps(gt_data, indent=2))
        print(f"\nTest image saved to: {test_image_path}")
    else:
        print("Image generation failed.") 