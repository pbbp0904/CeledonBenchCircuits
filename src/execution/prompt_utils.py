import os
import json

# Note: Prompt templates are less critical now as the main instruction
# comes directly from the test case JSON. This is kept simple.

# Import the constants from the generator module
# Assuming the script is run from the project root or src is in PYTHONPATH
try:
    from src.generation.image_generator import SHAPE_TYPES, COLORS
except ImportError:
    # Fallback if running script directly/path issues
    print("Warning: Could not import constants from image_generator. Prompt hints will be unavailable.")
    SHAPE_TYPES = ["circle", "square", "triangle", "rectangle", "star", "pentagon"]
    COLORS = {
        'red': (255, 0, 0), 'green': (0, 255, 0), 'blue': (0, 0, 255),
        'yellow': (255, 255, 0), 'magenta': (255, 0, 255), 'cyan': (0, 255, 255),
        # Add more if needed
    }

DEFAULT_INSTRUCTIONS = {
    "identify_all": "Analyze the provided image. For each numbered element, identify the shape it points to, its color, and its type. Return the result as a structured JSON object mapping each number (as a string key) to an object containing 'shape_type', 'color', and 'shape_id'.",
    # Add other task types here if needed
}

def format_prompt(test_case: dict) -> str:
    """Formats the prompt for the LLM, including context about valid shapes/colors."""
    instruction_template = test_case.get("instruction", DEFAULT_INSTRUCTIONS.get(test_case.get("task_type", ""), "Analyze the image."))
    n_value = test_case.get("n_value")

    # Add hints about valid shapes and colors based on N
    if n_value is not None and n_value > 0:
        num_shapes = min(n_value, len(SHAPE_TYPES))
        num_colors = min(n_value, len(COLORS))
        
        valid_shapes = list(SHAPE_TYPES)[:num_shapes]
        valid_colors = list(COLORS.keys())[:num_colors]
        
        shape_hint = f"Valid shape types for this image are: {valid_shapes}."
        color_hint = f"Valid color names for this image are: {valid_colors}."
        
        # Inject hints and fallback instruction
        instruction = (
            f"{instruction_template}\n\n"
            f"{shape_hint}\n"
            f"{color_hint}\n\n"
            f"Use only these exact shape types and color names in your JSON response.\n\n"
            f"If you are unable to see or analyze an image associated with this request, please respond with the JSON object: 'error': 'Image/Shapes not detected'."
        )
    else:
        instruction = instruction_template # No N-value, use original instruction

    # For this benchmark, the prompt is just the enhanced instruction.
    # The image is passed separately.
    prompt = instruction
    return prompt


# Example usage (optional, for testing)
if __name__ == '__main__':
    example_case = {
        "test_case_id": "n3_001_identify_all",
        "n_value": 3,
        "task_type": "identify_all",
        "instruction": "Analyze the provided image. For each numbered element, identify the shape it points to, the shape's color, and the shape's type. Return the result as a structured JSON object mapping each number (as a string key) to an object containing 'shape_type', 'color', and 'shape_id'.",
        "image_path": "n3/images/n3_001_identify_all.png",
        "expected_outcome": {},
        "evaluation_method": "detailed_comparison"
    }
    formatted_p = format_prompt(example_case)
    print("--- Example Formatted Prompt (N=3) ---")
    print(formatted_p)
    print("-" * 40)
    
    example_case_no_n = {
        "test_case_id": "unknown_001_identify_all",
        "task_type": "identify_all",
         "instruction": "Analyze this image and describe connections.",
        "image_path": "some/image.png",
    }
    formatted_p_no_n = format_prompt(example_case_no_n)
    print("--- Example Formatted Prompt (No N) ---")
    print(formatted_p_no_n)
    print("-" * 40) 