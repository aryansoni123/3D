import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union

import google.generativeai as genai
from dotenv import load_dotenv
from PIL import Image

from geometry_solver import solve_geometry


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


def build_semantic_prompt(regions_data: Dict[str, Any], front: Dict[str, Any], top: Dict[str, Any], side: Dict[str, Any]) -> str:
    """
    Build a prompt for Gemini to classify regions semantically.
    
    Geometry is already solved - LLM only needs to determine:
    - operation: "extrude" or "cut"
    - role: descriptive name (e.g., "table_top", "leg", "hole")
    - height_label: "tall", "short", "flush" (optional, for step detection)
    """
    
    regions = regions_data.get("regions", [])
    
    # Example output schema - LLM only provides semantics
    target_schema_example = {
        "features": [
            {
                "region_id": "region_0",
                "operation": "extrude",
                "role": "table_top",
                "height_label": "tall"
            },
            {
                "region_id": "region_1",
                "operation": "extrude",
                "role": "table_leg_block",
                "height_label": "short"
            }
        ]
    }
    
    instructions = f"""
You are a CAD geometry semantic classifier. Your job is to understand the INTENT behind geometric regions.

IMPORTANT: All geometric dimensions have already been computed from the 2D views using a priority system:
- X, Y positions and dimensions come from TOP view
- Z (height) comes from SIDE/FRONT views

You are provided with:
1. Pre-computed geometric regions (with exact x, y, width, depth, z_min, z_max)
2. Original 2D view images (front, top, side)
3. Original 2D view JSON data

Your task is to classify each region semantically:

For each region, determine:
- "operation": "extrude" (adds material) or "cut" (removes material)
- "role": Descriptive name like "table_top", "leg", "platform", "hole", "cavity", etc.
- "height_label": "tall" (full height), "short" (reduced height/step), "flush" (no height), or null

RULES:
- Look at the images to understand what each region represents
- Most regions are "extrude" (solid material)
- "cut" is for holes, cavities, or removed material (usually smaller, inside another region)
- "height_label" helps identify stepped/leveled surfaces
- Use the images to understand context - is this a table? A platform? A block with holes?

OUTPUT FORMAT - STRICTLY JSON ONLY:

{json.dumps(target_schema_example, indent=2)}

REQUIREMENTS:
- Output one entry per region_id
- Do NOT include any geometric dimensions (x, y, width, depth, height) - those are already computed
- Do NOT include position - that's already computed
- Only provide: region_id, operation, role, height_label
- Output ONLY JSON. NO EXPLANATIONS. NO MARKDOWN.
"""
    
    regions_text = f"""
PRE-COMPUTED GEOMETRIC REGIONS:
{json.dumps(regions_data, indent=2)}

ORIGINAL 2D VIEW DATA (for reference):
Front View: {json.dumps(front, indent=2)}
Top View: {json.dumps(top, indent=2)}
Side View: {json.dumps(side, indent=2)}
"""
    
    prompt = instructions + "\n\n" + regions_text
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


def merge_geometry_and_semantics(
    regions_data: Dict[str, Any],
    semantics_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge pre-computed geometry with LLM semantic classification into final 3D feature JSON.
    
    Args:
        regions_data: Output from geometry_solver.solve_geometry()
        semantics_data: Output from LLM (region classifications)
    
    Returns:
        Final 3D feature JSON compatible with house_generator.generate_model()
    """
    regions = regions_data.get("regions", [])
    semantic_features = semantics_data.get("features", [])
    
    # Create lookup map for semantics
    semantics_map = {}
    for feat in semantic_features:
        region_id = feat.get("region_id")
        if region_id:
            semantics_map[region_id] = feat
    
    # Build final features
    final_features = []
    used_ids = set()  # Track used IDs to ensure uniqueness
    
    for region in regions:
        region_id = region.get("id")
        semantic = semantics_map.get(region_id, {})
        
        # Defaults if LLM didn't classify
        operation = semantic.get("operation", "extrude")
        role = semantic.get("role", region_id)
        height_label = semantic.get("height_label")
        
        # Ensure unique ID
        base_id = role
        feature_id = base_id
        counter = 1
        while feature_id in used_ids:
            feature_id = f"{base_id}_{counter}"
            counter += 1
        used_ids.add(feature_id)
        
        # Extract geometry
        x = region.get("x", 0.0)
        y = region.get("y", 0.0)
        width = region.get("width", 0.0)
        depth = region.get("depth", 0.0)
        z_min = region.get("z_min", 0.0)
        z_max = region.get("z_max", 0.0)
        base_height = region.get("height", z_max - z_min)
        
        # Adjust height based on height_label if provided
        # But preserve the geometry solver's computed height as default
        height = base_height
        
        if height_label == "short" and len(regions) > 1:
            # Find the tallest region
            max_height = max(r.get("height", 0) for r in regions)
            # Use a reasonable fraction (could be improved with better step detection)
            # For now, use 50% of max height
            height = max_height * 0.5
        elif height_label == "flush":
            height = 0.0
        elif height_label == "tall":
            # Use full height (already set)
            height = base_height
        # If height_label is None or unknown, use base_height (already set)
        
        # Build feature
        feature = {
            "id": feature_id,
            "type": operation,
            "profile": {
                "shape": "rectangle",
                "width": width,
                "height": height  # Profile height (Z dimension)
            },
            "dimensions": {
                "width": width,
                "depth": depth,
                "height": height
            },
            "position": {
                "x": x,
                "y": y,
                "z": z_min
            },
            "source_region": region_id
        }
        
        final_features.append(feature)
    
    return {
        "coordinate_system": {
            "origin": {"x": 0, "y": 0, "z": 0},
            "axes": {
                "front_view": {"horizontal": "x", "vertical": "z"},
                "top_view": {"horizontal": "x", "vertical": "y"},
                "side_view": {"horizontal": "y", "vertical": "z"}
            }
        },
        "features": final_features
    }


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

    print("=" * 60)
    print("2D → 3D RECONSTRUCTION PIPELINE")
    print("=" * 60)
    
    # STEP 1: Solve geometry deterministically (priority-based)
    print("\n[Step 1/3] Solving geometry from 2D views...")
    try:
        regions_data = solve_geometry(args.top, args.side, args.front)
        print(f"✓ Extracted {len(regions_data['regions'])} regions")
        print(f"  X/Y source: {regions_data['metadata']['x_source']}, {regions_data['metadata']['y_source']}")
        print(f"  Z source: {regions_data['metadata']['z_source']}")
    except Exception as e:
        print(f"✗ Geometry solver failed: {e}")
        raise
    
    # STEP 2: LLM semantic classification
    print("\n[Step 2/3] Classifying regions semantically (LLM)...")
    
    # Load JSON files for LLM context
    front = load_json(args.front)
    top = load_json(args.top)
    side = load_json(args.side)
    
    # Load images for multimodal input
    base_dir = Path(args.front).parent if Path(args.front).parent != Path(".") else None
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
    
    # Build semantic prompt
    prompt = build_semantic_prompt(regions_data, front, top, side)
    
    if not images:
        print("Warning: No images found. Using text-only mode.")
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
        raw_text = call_gemini_multimodal(prompt, images, image_labels, args.model)
    
    print("Extracting semantic classifications from LLM response...")
    try:
        semantics_json = extract_json_from_text(raw_text)
        print(f"✓ Classified {len(semantics_json.get('features', []))} regions")
    except Exception as e:
        print(f"✗ Failed to parse LLM response: {e}")
        print(f"Raw response: {raw_text[:500]}...")
        raise
    
    # STEP 3: Merge geometry + semantics
    print("\n[Step 3/3] Merging geometry and semantics...")
    try:
        final_features_json = merge_geometry_and_semantics(regions_data, semantics_json)
        print(f"✓ Generated {len(final_features_json['features'])} final features")
        
        # Debug: Print first feature details
        if final_features_json['features']:
            feat = final_features_json['features'][0]
            print(f"\n  Sample feature '{feat['id']}':")
            print(f"    Position: {feat['position']}")
            print(f"    Dimensions: {feat['dimensions']}")
            print(f"    Type: {feat['type']}")
    except Exception as e:
        print(f"✗ Merge failed: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # Save output
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(final_features_json, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"✓ SUCCESS: Saved 3D feature JSON to {args.output}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

