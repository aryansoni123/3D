import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import google.generativeai as genai
import numpy as np
import trimesh
from dotenv import load_dotenv
from PIL import Image


GEMINI_MODEL_NAME_DEFAULT = "gemini-flash-latest"


def load_image(path: str) -> Image.Image:
    img_path = Path(path)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        return Image.open(img_path)
    except Exception as e:
        raise RuntimeError(f"Failed to load image {path}: {e}") from e


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


def build_prompt() -> str:
    """
    Build the instruction text for Gemini.

    We let the model choose primitives and structure of the object, but
    we enforce a strict JSON schema for the final output so that we can
    reliably build a mesh with trimesh.
    """
    target_schema_example = {
        "primitives": [
            {
                "id": "block_0",
                "kind": "box",
                "operation": "add",
                "size": {
                    "width": 87.0,
                    "depth": 59.0,
                    "height": 87.0,
                },
                "position": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                },
            },
            {
                "id": "leg_left",
                "kind": "box",
                "operation": "add",
                "size": {
                    "width": 4.0,
                    "depth": 59.0,
                    "height": 87.0,
                },
                "position": {
                    "x": 0.0,
                    "y": 0.0,
                    "z": 0.0,
                },
            },
            {
                "id": "hole_0",
                "kind": "cylinder",
                "operation": "cut",
                "size": {
                    "radius": 5.0,
                    "height": 20.0,
                },
                "position": {
                    "x": 40.0,
                    "y": 0.0,
                    "z": 40.0,
                },
            },
        ]
    }

    instructions = f"""
You are a senior CAD/geometry engineer.

You are given three orthographic 2D views (front, top, side) of a single rigid 3D object.
Your job is to imagine a clean 3D model that best matches these views, and then describe it
as a small set of simple primitives that can be built with trimesh.

Important:
- Do NOT try to perfectly match every pixel.
- Instead, infer a clean, idealized 3D model that captures the main geometry.

You may use these primitives:
- "box"       (rectangular block)
- "cylinder"  (round column / hole)
- "sphere"    (ball or rounded feature)

Each primitive must have:
- "id": unique string identifier
- "kind": "box" | "cylinder" | "sphere"
- "operation": "add" (add material) or "cut" (remove material)
- "size": dimensions of the primitive
  - If kind == "box":
      {{"width": <X-size>, "depth": <Y-size>, "height": <Z-size>}}
  - If kind == "cylinder":
      {{"radius": <radius>, "height": <Y-size>}}   # axis along Y
  - If kind == "sphere":
      {{"radius": <radius>}}
- "position": origin of the primitive in 3D space (bottom-left-front corner for boxes)
  - {{"x": <X>, "y": <Y>, "z": <Z>}}

Coordinate system:
- X: left-right (width)
- Y: depth (back-front)
- Z: vertical (height)

You MUST output STRICTLY VALID JSON with this schema:

{json.dumps(target_schema_example, indent=2)}

Rules:
- Use only a small number of primitives (3–10) that best approximate the object.
- Use floating-point numbers for sizes and positions.
- "operation": use "add" for solid parts, "cut" for holes/cavities.
- Do NOT include any explanation, comments, or markdown. Output ONLY a JSON object.
"""

    return instructions


def call_gemini_multimodal(
    front_img: Image.Image,
    top_img: Image.Image,
    side_img: Image.Image,
    model_name: str,
) -> str:
    """
    Call Gemini with multimodal input: three images + text prompt.
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

    prompt = build_prompt()

    content_parts: List[Any] = []

    if front_img is not None:
        content_parts.append("Front view:")
        content_parts.append(front_img)
    if top_img is not None:
        content_parts.append("Top view:")
        content_parts.append(top_img)
    if side_img is not None:
        content_parts.append("Side view:")
        content_parts.append(side_img)

    content_parts.append(prompt)

    response = model.generate_content(content_parts)
    text = response.text or ""
    return text.strip()


def build_mesh_from_primitives(data: Dict[str, Any]) -> trimesh.Trimesh:
    """
    Build a mesh from LLM-defined primitives.
    """
    primitives = data.get("primitives", [])
    if not primitives:
        raise ValueError("No primitives found in LLM output.")

    final_mesh: trimesh.Trimesh | None = None

    for prim in primitives:
        prim_id = prim.get("id", "primitive")
        kind = prim.get("kind")
        operation = prim.get("operation", "add")
        size = prim.get("size", {})
        pos = prim.get("position", {})

        if kind not in ["box", "cylinder", "sphere"]:
            print(f"Warning: Primitive '{prim_id}' has unsupported kind '{kind}', skipping.")
            continue

        try:
            if kind == "box":
                width = float(size.get("width", 0))
                depth = float(size.get("depth", 0))
                height = float(size.get("height", 0))
                if width <= 0 or depth <= 0 or height <= 0:
                    print(f"Warning: Box '{prim_id}' has non-positive size {size}, skipping.")
                    continue
                mesh = trimesh.creation.box(extents=(width, depth, height))

            elif kind == "cylinder":
                radius = float(size.get("radius", 0))
                height = float(size.get("height", 0))
                if radius <= 0 or height <= 0:
                    print(f"Warning: Cylinder '{prim_id}' has non-positive size {size}, skipping.")
                    continue
                # Cylinder axis along Y
                mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=64)

            else:  # sphere
                radius = float(size.get("radius", 0))
                if radius <= 0:
                    print(f"Warning: Sphere '{prim_id}' has non-positive radius {size}, skipping.")
                    continue
                mesh = trimesh.creation.icosphere(radius=radius, subdivisions=3)

            # Position
            translation = np.array(
                [
                    float(pos.get("x", 0.0)),
                    float(pos.get("y", 0.0)),
                    float(pos.get("z", 0.0)),
                ]
            )
            mesh.apply_translation(translation)

            # Combine
            if final_mesh is None:
                final_mesh = mesh
            else:
                if operation == "cut":
                    try:
                        final_mesh = final_mesh.difference(mesh)
                    except Exception as e:
                        print(f"Warning: Failed to subtract '{prim_id}': {e}")
                else:
                    try:
                        final_mesh = final_mesh.union(mesh)
                    except Exception as e:
                        print(f"Warning: Failed to union '{prim_id}': {e}")

        except Exception as e:
            print(f"Warning: Failed to process primitive '{prim_id}': {e}")
            continue

    if final_mesh is None:
        raise ValueError("No valid primitives were processed. Final mesh is empty.")

    return final_mesh


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fully LLM-based 2D→3D reconstruction: "
            "use three images (front, top, side) and Gemini to generate a "
            "primitive list, then build a mesh with trimesh."
        )
    )
    parser.add_argument(
        "--front-img",
        type=str,
        required=True,
        help="Path to front view image.",
    )
    parser.add_argument(
        "--top-img",
        type=str,
        required=True,
        help="Path to top view image.",
    )
    parser.add_argument(
        "--side-img",
        type=str,
        required=True,
        help="Path to side view image.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="llm_model.stl",
        help="Output mesh file (default: llm_model.stl)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=GEMINI_MODEL_NAME_DEFAULT,
        help="Gemini model name to use.",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("FULLY LLM-BASED 2D → 3D RECONSTRUCTION")
    print("=" * 60)

    # Load images
    print("\n[Step 1/3] Loading images...")
    front_img = load_image(args.front_img)
    top_img = load_image(args.top_img)
    side_img = load_image(args.side_img)
    print("✓ Images loaded.")

    # Call Gemini
    print("\n[Step 2/3] Calling Gemini to generate primitives...")
    raw_text = call_gemini_multimodal(front_img, top_img, side_img, args.model)

    print("Extracting JSON from LLM response...")
    try:
        primitives_json = extract_json_from_text(raw_text)
        print(f"✓ Parsed {len(primitives_json.get('primitives', []))} primitives.")
    except Exception as e:
        print(f"✗ Failed to parse LLM response: {e}")
        print(raw_text[:1000])
        raise

    # Build mesh
    print("\n[Step 3/3] Building mesh from primitives...")
    mesh = build_mesh_from_primitives(primitives_json)
    print("✓ Mesh built.")

    # Export
    output_path = args.output
    mesh.export(output_path)
    print(f"\n✓ Exported mesh to {output_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()

