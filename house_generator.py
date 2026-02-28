import argparse
import json
import sys
from pathlib import Path

import numpy as np
import trimesh
from shapely.geometry import Polygon


# -----------------------------
# Create 3D mesh from 2D profile
# -----------------------------
def create_extruded_mesh(profile, depth):
    """
    Create a 3D mesh by extruding a 2D profile.

    Args:
        profile: Dictionary with "shape" key and shape-specific parameters
        depth: Extrusion depth along Y-axis

    Returns:
        trimesh.Trimesh: The generated 3D mesh
    """
    shape = profile.get("shape")
    if not shape:
        raise ValueError("Profile must have a 'shape' field")

    if shape == "rectangle":
        width = profile.get("width")
        height = profile.get("height")
        if width is None or height is None:
            raise ValueError("Rectangle profile must have 'width' and 'height'")
        # trimesh box: extents are (width, depth, height) in (X, Y, Z)
        mesh = trimesh.creation.box(extents=(width, depth, height))
        return mesh

    elif shape == "circle":
        radius = profile.get("radius")
        if radius is None:
            raise ValueError("Circle profile must have 'radius'")
        # trimesh cylinder: height is along Y-axis by default
        mesh = trimesh.creation.cylinder(radius=radius, height=depth, sections=64)
        return mesh

    elif shape == "polygon":
        points = profile.get("points")
        if not points:
            raise ValueError("Polygon profile must have 'points' list")
        if len(points) < 3:
            raise ValueError("Polygon must have at least 3 points")
        # Convert to Shapely polygon and extrude
        polygon = Polygon(points)
        if not polygon.is_valid:
            raise ValueError(f"Invalid polygon: {polygon}")
        # Extrude polygon along Y axis (depth)
        mesh = trimesh.creation.extrude_polygon(polygon, depth)
        return mesh

    else:
        raise ValueError(f"Unsupported shape type: {shape}. Supported: rectangle, circle, polygon")


# -----------------------------
# Apply position transform
# -----------------------------
def apply_position(mesh, position):
    """
    Translate a mesh to a specific position.

    Args:
        mesh: trimesh.Trimesh object
        position: Dictionary with optional "x", "y", "z" keys (defaults to 0)

    Returns:
        trimesh.Trimesh: The translated mesh (modified in-place)
    """
    if position is None:
        position = {}
    translation = np.array([
        position.get("x", 0),
        position.get("y", 0),
        position.get("z", 0)
    ])
    mesh.apply_translation(translation)
    return mesh


# -----------------------------
# Main Feature Engine
# -----------------------------
def generate_model(data):
    """
    Generate a 3D mesh from a feature-based JSON description.

    Supports two formats:
    1. Old format: features have "depth" field
    2. New format: features have "dimensions" object with "width", "depth", "height"

    Args:
        data: Dictionary with "features" list. Each feature has:
            - "id": Unique identifier (optional)
            - "type": "extrude" (add) or "cut" (subtract)
            - "profile": 2D shape definition (rectangle/circle/polygon)
            - "depth": Extrusion depth (old format) OR
            - "dimensions": {"width": ..., "depth": ..., "height": ...} (new format)
            - "position": {"x": ..., "y": ..., "z": ...}

    Returns:
        trimesh.Trimesh: The final combined 3D mesh
    """
    if "features" not in data:
        raise ValueError("Input data must have a 'features' field")

    features = data["features"]
    if not features:
        raise ValueError("Features list is empty")

    final_mesh = None

    for idx, feature in enumerate(features):
        feature_id = feature.get("id", f"feature_{idx}")
        feature_type = feature.get("type")

        if feature_type not in ["extrude", "cut"]:
            raise ValueError(
                f"Feature '{feature_id}': Unsupported type '{feature_type}'. "
                "Must be 'extrude' or 'cut'"
            )

        profile = feature.get("profile")
        if not profile:
            raise ValueError(f"Feature '{feature_id}': Missing 'profile' field")

        # Support both old format (depth) and new format (dimensions)
        dimensions = feature.get("dimensions")
        if dimensions:
            # New format: use dimensions.depth
            depth = dimensions.get("depth")
            if depth is None:
                raise ValueError(f"Feature '{feature_id}': dimensions object missing 'depth' field")
        else:
            # Old format: use top-level depth
            depth = feature.get("depth")
            if depth is None:
                raise ValueError(
                    f"Feature '{feature_id}': Missing 'depth' field. "
                    "Either provide 'depth' (old format) or 'dimensions.depth' (new format)."
                )

        if depth <= 0:
            print(f"Warning: Feature '{feature_id}' has non-positive depth {depth}, skipping")
            continue

        position = feature.get("position", {})

        try:
            mesh = create_extruded_mesh(profile, depth)
            mesh = apply_position(mesh, position)
        except Exception as e:
            raise RuntimeError(f"Error processing feature '{feature_id}': {e}") from e

        if feature_type == "extrude":
            if final_mesh is None:
                final_mesh = mesh
            else:
                try:
                    final_mesh = final_mesh.union(mesh)
                except Exception as e:
                    print(f"Warning: Failed to union feature '{feature_id}': {e}")
                    # Continue with next feature

        elif feature_type == "cut":
            if final_mesh is None:
                raise ValueError(
                    f"Feature '{feature_id}': Cannot apply cut before base solid exists. "
                    "Ensure at least one 'extrude' feature comes before any 'cut' features."
                )
            try:
                final_mesh = final_mesh.difference(mesh)
            except Exception as e:
                print(f"Warning: Failed to subtract feature '{feature_id}': {e}")
                # Continue with next feature

    if final_mesh is None:
        raise ValueError("No valid features were processed. Final mesh is empty.")

    return final_mesh


# -----------------------------
# Load JSON from file
# -----------------------------
def load_json_file(json_path: str) -> dict:
    """
    Load and parse a JSON file.

    Args:
        json_path: Path to JSON file

    Returns:
        dict: Parsed JSON data
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in file '{json_path}': {e}") from e


# -----------------------------
# Export mesh to file
# -----------------------------
def export_mesh(mesh: trimesh.Trimesh, output_path: str) -> None:
    """
    Export a mesh to a file. Format is determined by file extension.

    Args:
        mesh: trimesh.Trimesh object to export
        output_path: Output file path (supports .stl, .obj, .ply, etc.)
    """
    path = Path(output_path)
    suffix = path.suffix.lower()

    supported_formats = [".stl", ".obj", ".ply", ".off", ".dae", ".glb"]
    if suffix not in supported_formats:
        print(f"Warning: Unknown file extension '{suffix}'. Trying default export...")

    try:
        mesh.export(output_path)
        print(f"✓ Mesh exported successfully to: {output_path}")
        print(f"  Vertices: {len(mesh.vertices)}, Faces: {len(mesh.faces)}")
    except Exception as e:
        raise RuntimeError(f"Failed to export mesh to '{output_path}': {e}") from e


# -----------------------------
# Main CLI Entry Point
# -----------------------------
def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a 3D mesh from a feature-based JSON file. "
            "The JSON should contain a 'features' array with extrude/cut operations."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example JSON structure:
{
  "features": [
    {
      "id": "base",
      "type": "extrude",
      "profile": {"shape": "rectangle", "width": 100, "height": 70},
      "depth": 80,
      "position": {"x": 0, "y": 0, "z": 0}
    },
    {
      "id": "hole",
      "type": "cut",
      "profile": {"shape": "circle", "radius": 10},
      "depth": 20,
      "position": {"x": 50, "y": 0, "z": 50}
    }
  ]
}
        """,
    )
    parser.add_argument(
        "input_json",
        type=str,
        nargs="?",
        default="model_features.json",
        help="Path to input JSON file with feature definitions (default: model_features.json)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="Path to output mesh file (default: input name with .stl extension)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["stl", "obj", "ply", "off", "dae", "glb"],
        default="stl",
        help="Output file format (default: stl)",
    )

    args = parser.parse_args()

    # Load JSON
    try:
        print(f"Loading features from: {args.input_json}")
        data = load_json_file(args.input_json)
        print(f"✓ Loaded {len(data.get('features', []))} features")
    except Exception as e:
        print(f"Error loading JSON file: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate mesh
    try:
        print("\nGenerating 3D mesh...")
        mesh = generate_model(data)
        print("✓ Mesh generation completed")
    except Exception as e:
        print(f"Error generating mesh: {e}", file=sys.stderr)
        sys.exit(1)

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        input_path = Path(args.input_json)
        output_path = input_path.with_suffix(f".{args.format}")

    # Export mesh
    try:
        export_mesh(mesh, str(output_path))
    except Exception as e:
        print(f"Error exporting mesh: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()