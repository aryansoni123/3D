import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image


GEMINI_MODEL_NAME_DEFAULT = "gemini-flash-latest"


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_image_from_json(json_data: Dict[str, Any], base_dir: Path = None) -> Union[Image.Image, None]:
    """
    Load the image file referenced in the JSON's image_path field.
    
    Args:
        json_data: The view JSON with "image_path" field
        base_dir: Base directory to resolve relative paths (default: current dir)
    
    Returns:
        PIL Image or None if not found
    """
    image_path = json_data.get("image_path")
    if not image_path:
        return None
    
    if base_dir:
        full_path = base_dir / image_path
    else:
        full_path = Path(image_path)
    
    if not full_path.exists():
        # Try to find the image in common locations
        for possible_dir in [Path("."), Path("Ref Image"), Path("2D Json")]:
            test_path = possible_dir / Path(image_path).name
            if test_path.exists():
                full_path = test_path
                break
        else:
            print(f"Warning: Image not found: {image_path}")
            return None
    
    try:
        return Image.open(full_path)
    except Exception as e:
        print(f"Warning: Failed to load image {full_path}: {e}")
        return None


def build_prompt(front: Dict[str, Any], top: Dict[str, Any], side: Dict[str, Any]) -> str:
    """
    Build a text-only prompt for Gemini that explains the task and embeds
    the OpenCV-extracted 2D geometry for all three views.
    """
    # Precise schema with coordinate system and dimension extraction rules
    target_schema_example = {
        "coordinate_system": {
            "origin": {"x": 0, "y": 0, "z": 0},
            "axes": {
                "front_view": {"horizontal": "x", "vertical": "z"},
                "top_view": {"horizontal": "x", "vertical": "y"},
                "side_view": {"horizontal": "y", "vertical": "z"}
            }
        },
        "features": [
            {
                "id": "base_block",
                "type": "extrude",
                "profile": {
                    "shape": "rectangle",
                    "width": 100.0,
                    "height": 70.0
                },
                "dimensions": {
                    "width": 100.0,
                    "depth": 80.0,
                    "height": 70.0
                },
                "position": {"x": 0.0, "y": 0.0, "z": 0.0},
                "source_shapes": {
                    "front_view": ["rect_0"],
                    "top_view": ["rect_0"],
                    "side_view": ["rect_0"]
                }
            },
            {
                "id": "hole_cut",
                "type": "cut",
                "profile": {
                    "shape": "circle",
                    "radius": 10.0
                },
                "dimensions": {
                    "width": 20.0,
                    "depth": 5.0,
                    "height": 20.0
                },
                "position": {"x": 50.0, "y": 0.0, "z": 50.0},
                "source_shapes": {
                    "front_view": ["circle_1"],
                    "top_view": ["circle_1"]
                }
            }
        ]
    }

    # Step-by-step instructions with explicit coordinate extraction
    instructions = f"""
You are a precision CAD geometry reconstruction engine. Reconstruct 3D features from orthographic views.

STEP-BY-STEP PROCESS:

STEP 1: UNDERSTAND COORDINATE SYSTEMS
- Front view shows: X (left-right) and Z (bottom-top)
- Top view shows: X (left-right) and Y (back-front, depth)
- Side view shows: Y (left-right, depth) and Z (bottom-top)

STEP 2: EXTRACT BASE DIMENSIONS FROM EACH VIEW

For FRONT VIEW (X-Z plane):
- Find the main rectangle's params: x, y, width, height
- X range = [x, x+width]
- Z range = [y, y+height]  (y is bottom, y+height is top)
- Note: This gives you WIDTH (X) and HEIGHT (Z)

For TOP VIEW (X-Y plane):
- Find the main rectangle's params: x, y, width, height  
- X range = [x, x+width]
- Y range = [y, y+height]  (y is back, y+height is front)
- Note: This gives you WIDTH (X) and DEPTH (Y)

For SIDE VIEW (Y-Z plane):
- Find the main rectangle's params: x, y, width, height
- Y range = [x, x+width]  (x represents Y in side view!)
- Z range = [y, y+height]  (y is bottom, y+height is top)
- Note: This gives you DEPTH (Y) and HEIGHT (Z)

STEP 3: ALIGN COORDINATE SYSTEMS
- The views may have different origins (different x,y starting points)
- To find the TRUE 3D position, use the MINIMUM values:
  - position.x = min(front_view.x, top_view.x)
  - position.y = min(top_view.y, side_view.x)  (side_view.x is Y!)
  - position.z = min(front_view.y, side_view.y)  (both are Z bottom)

STEP 4: CALCULATE 3D DIMENSIONS
- dimensions.width = X span = max(front_view.x+width, top_view.x+width) - position.x
- dimensions.depth = Y span = max(top_view.y+height, side_view.x+width) - position.y
- dimensions.height = Z span = max(front_view.y+height, side_view.y+height) - position.z

STEP 5: HANDLE INTERIOR LINES (SPLITS)
- Vertical lines in FRONT view (constant X) = split along X-axis (different Y or Z regions)
- Horizontal lines in TOP view (constant Y) = split along Y-axis (different Z levels)
- Horizontal lines in SIDE view (constant Z) = split along Z-axis (different X or Y regions)
- When you see a split line, create SEPARATE features for each region
- Each region gets its own position and dimensions

STEP 6: MATCH SHAPES ACROSS VIEWS
- A 3D feature appears in ALL THREE views
- Match by checking if X ranges overlap (front & top), Y ranges overlap (top & side), Z ranges overlap (front & side)
- If a shape only appears in 1-2 views, it might be a cut or a partial feature

STEP 7: DETERMINE FEATURE TYPE
- "extrude": Solid material (most features)
- "cut": Hole, cavity, or removed material (usually smaller, inside another shape)

CRITICAL RULES:
1. position.x/y/z MUST be the minimum corner (bottom-left-back in 3D space)
2. dimensions MUST be positive and match across views where they should
3. If dimensions don't match, use the LARGER value (some views may show partial features)
4. Interior lines create SEPARATE features - don't merge them
5. Use EXACT values from JSON params, not approximate

OUTPUT FORMAT - STRICTLY JSON ONLY:

{json.dumps(target_schema_example, indent=2)}

VALIDATION CHECKLIST before outputting:
✓ position.x = minimum X from all views
✓ position.y = minimum Y from top/side views  
✓ position.z = minimum Z from front/side views
✓ dimensions.width matches X span
✓ dimensions.depth matches Y span
✓ dimensions.height matches Z span
✓ Each feature has unique id
✓ source_shapes references correct shape IDs from JSON

OUTPUT ONLY JSON. NO EXPLANATIONS. NO MARKDOWN.
"""

    # Extract key values for easier reference
    def extract_key_values(view_data, view_name):
        shapes = view_data.get("shapes", [])
        lines = view_data.get("lines", [])
        summary = {"view": view_name, "shapes": [], "lines": []}
        
        for shape in shapes:
            if shape.get("type") == "rectangle":
                params = shape.get("params", {})
                summary["shapes"].append({
                    "id": shape.get("id"),
                    "x": params.get("x"),
                    "y": params.get("y"),
                    "width": params.get("width"),
                    "height": params.get("height"),
                    "x_range": [params.get("x"), params.get("x", 0) + params.get("width", 0)],
                    "y_range": [params.get("y"), params.get("y", 0) + params.get("height", 0)]
                })
        
        for line in lines:
            summary["lines"].append({
                "id": line.get("id"),
                "start": line.get("start"),
                "end": line.get("end"),
                "is_vertical": abs(line.get("start", [0,0])[0] - line.get("end", [0,0])[0]) < 2,
                "is_horizontal": abs(line.get("start", [0,0])[1] - line.get("end", [0,0])[1]) < 2
            })
        
        return summary
    
    front_summary = extract_key_values(front, "front")
    top_summary = extract_key_values(top, "top")
    side_summary = extract_key_values(side, "side")
    
    summary_text = f"""
EXTRACTED VALUES FROM JSON (for reference):

FRONT VIEW (X-Z plane):
{json.dumps(front_summary, indent=2)}

TOP VIEW (X-Y plane):
{json.dumps(top_summary, indent=2)}

SIDE VIEW (Y-Z plane):
{json.dumps(side_summary, indent=2)}

FULL JSON DATA:
{json.dumps({"front_view": front, "top_view": top, "side_view": side}, indent=2)}
"""
    
    prompt = instructions + "\n\n" + summary_text
    return prompt


def call_gemini_multimodal(
    prompt: str, 
    images: List[Image.Image], 
    image_labels: List[str],
    model_name: str
) -> str:
    """
    Call Gemini with multimodal input (images + text).
    
    Args:
        prompt: Text prompt/instructions
        images: List of PIL Images to send
        image_labels: Labels for each image (e.g., ["front_view", "top_view", "side_view"])
        model_name: Gemini model name
    
    Returns:
        Response text from Gemini
    """
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it to your Gemini API key before running this script."
        )

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    
    # Build content list: alternate images and text
    content_parts = []
    
    # Add images with labels
    for img, label in zip(images, image_labels):
        if img is not None:
            content_parts.append(f"Image: {label}")
            content_parts.append(img)
    
    # Add the main prompt at the end
    content_parts.append(prompt)
    
    try:
        response = model.generate_content(content_parts)
        text = response.text or ""
        return text.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini API call failed: {e}") from e


def extract_json_from_text(text: str) -> Dict[str, Any]:
    """
    Robustly extract a JSON object from model text output that should
    contain only JSON, but might occasionally be wrapped.
    """
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("Model output did not contain a JSON object.")
    json_str = text[start : end + 1]
    return json.loads(json_str)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Combine OpenCV-extracted 2D view JSONs (front, top, side) "
            "and call Gemini to reconstruct 3D-ready feature JSON "
            "compatible with house_generator.generate_model."
        )
    )
    parser.add_argument(
        "--front",
        type=str,
        default="2D Json/front_view.json",
        help="Path to front-view JSON file.",
    )
    parser.add_argument(
        "--top",
        type=str,
        default="2D Json/top_view.json",
        help="Path to top-view JSON file.",
    )
    parser.add_argument(
        "--side",
        type=str,
        default="2D Json/side_view.json",
        help="Path to side-view JSON file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="model_features.json",
        help="Path to output 3D feature JSON file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=GEMINI_MODEL_NAME_DEFAULT,
        help="Gemini model name to use.",
    )

    args = parser.parse_args()

    # Determine base directory for resolving image paths
    base_dir = Path(args.front).parent if Path(args.front).parent != Path(".") else None

    # Load JSON files
    print("Loading 2D view JSON files...")
    front = load_json(args.front)
    top = load_json(args.top)
    side = load_json(args.side)

    # Load actual images for multimodal input
    print("Loading images for multimodal analysis...")
    front_img = load_image_from_json(front, base_dir)
    top_img = load_image_from_json(top, base_dir)
    side_img = load_image_from_json(side, base_dir)
    
    images = [img for img in [front_img, top_img, side_img] if img is not None]
    image_labels = []
    if front_img:
        image_labels.append("front_view")
    if top_img:
        image_labels.append("top_view")
    if side_img:
        image_labels.append("side_view")
    
    if not images:
        print("Warning: No images found. Falling back to text-only mode.")
        # Fallback to text-only
        prompt = build_prompt(front, top, side)
        # Use a simpler text-only call
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(args.model)
        response = model.generate_content(prompt)
        raw_text = response.text or ""
    else:
        print(f"Using multimodal input with {len(images)} images")
        prompt = build_prompt(front, top, side)
        raw_text = call_gemini_multimodal(prompt, images, image_labels, args.model)
    
    print("Extracting JSON from response...")
    features_json = extract_json_from_text(raw_text)

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(features_json, f, indent=2)

    print(f"✓ Saved 3D feature JSON to {args.output}")


if __name__ == "__main__":
    main()

