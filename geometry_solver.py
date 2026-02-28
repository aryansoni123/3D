"""
Priority-based geometry solver for 2D to 3D reconstruction.

View Priority:
- X (width): Top > Front > Side
- Y (depth): Top > Side > Front  
- Z (height): Side > Front > Top

This module extracts pure numeric geometry from 2D views without LLM.
"""
import json
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass


@dataclass
class Region:
    """A 3D region with all dimensions computed from priority-based extraction."""
    id: str
    x: float
    y: float
    width: float
    depth: float
    z_min: float
    z_max: float


def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_x_y_from_top(top_data: Dict[str, Any]) -> Tuple[float, float, float, float]:
    """
    Extract X, Y geometry from top view (priority 1 for X and Y).
    
    Returns: (x_min, y_min, width, depth)
    """
    shapes = top_data.get("shapes", [])
    if not shapes:
        raise ValueError("Top view has no shapes")
    
    # Find main rectangle (usually rect_0)
    main_rect = None
    for shape in shapes:
        if shape.get("type") == "rectangle":
            main_rect = shape
            break
    
    if not main_rect:
        raise ValueError("Top view has no rectangle")
    
    params = main_rect.get("params", {})
    x = params.get("x", 0.0)
    y = params.get("y", 0.0)
    width = params.get("width", 0.0)
    height = params.get("height", 0.0)  # This is depth in top view
    
    return x, y, width, height


def split_regions_by_lines(
    x_min: float, y_min: float, width: float, depth: float,
    lines: List[Dict[str, Any]]
) -> List[Region]:
    """
    Split the main footprint into sub-regions based on interior lines.
    
    Returns list of regions (if no splits, returns single region).
    """
    regions = []
    
    # Find split lines
    x_splits = []  # Vertical lines (constant X)
    y_splits = []  # Horizontal lines (constant Y)
    
    for line in lines:
        start = line.get("start", [0, 0])
        end = line.get("end", [0, 0])
        
        # Check if line is vertical (constant X, splits in Y)
        if abs(start[0] - end[0]) < 2:
            x_pos = (start[0] + end[0]) / 2
            if x_min < x_pos < x_min + width:
                x_splits.append(x_pos)
        
        # Check if line is horizontal (constant Y, splits in X)
        if abs(start[1] - end[1]) < 2:
            y_pos = (start[1] + end[1]) / 2
            if y_min < y_pos < y_min + depth:
                y_splits.append(y_pos)
    
    # Sort splits
    x_splits = sorted(set(x_splits))
    y_splits = sorted(set(y_splits))
    
    # Create regions from splits
    if not x_splits and not y_splits:
        # No splits - single region
        regions.append(Region(
            id="region_0",
            x=x_min,
            y=y_min,
            width=width,
            depth=depth,
            z_min=0.0,  # Will be filled later
            z_max=0.0
        ))
    else:
        # Create grid of regions
        x_boundaries = [x_min] + x_splits + [x_min + width]
        y_boundaries = [y_min] + y_splits + [y_min + depth]
        
        region_idx = 0
        for i in range(len(x_boundaries) - 1):
            for j in range(len(y_boundaries) - 1):
                regions.append(Region(
                    id=f"region_{region_idx}",
                    x=x_boundaries[i],
                    y=y_boundaries[j],
                    width=x_boundaries[i + 1] - x_boundaries[i],
                    depth=y_boundaries[j + 1] - y_boundaries[j],
                    z_min=0.0,
                    z_max=0.0
                ))
                region_idx += 1
    
    return regions


def extract_z_from_side_front(
    side_data: Dict[str, Any],
    front_data: Dict[str, Any],
    regions: List[Region]
) -> List[Region]:
    """
    Extract Z (height) dimensions from side/front views (priority: Side > Front).
    
    Updates regions with z_min and z_max.
    """
    # Priority 1: Side view
    side_shapes = side_data.get("shapes", [])
    side_z_min = None
    side_z_max = None
    
    if side_shapes:
        side_rect = None
        for shape in side_shapes:
            if shape.get("type") == "rectangle":
                side_rect = shape
                break
        
        if side_rect:
            params = side_rect.get("params", {})
            # In side view: x = Y (depth), y = Z (height)
            side_z_min = params.get("y", 0.0)
            side_height = params.get("height", 0.0)
            side_z_max = side_z_min + side_height
    
    # Priority 2: Front view (fallback if side view missing/invalid)
    front_shapes = front_data.get("shapes", [])
    front_z_min = None
    front_z_max = None
    
    if front_shapes:
        front_rect = None
        for shape in front_shapes:
            if shape.get("type") == "rectangle":
                front_rect = shape
                break
        
        if front_rect:
            params = front_rect.get("params", {})
            # In front view: x = X (width), y = Z (height)
            front_z_min = params.get("y", 0.0)
            front_height = params.get("height", 0.0)
            front_z_max = front_z_min + front_height
    
    # Use priority: Side > Front
    global_z_min = side_z_min if side_z_min is not None else front_z_min
    global_z_max = side_z_max if side_z_max is not None else front_z_max
    
    # If side view height seems wrong (too small), use front view
    if side_z_max and front_z_max:
        side_height = side_z_max - side_z_min
        front_height = front_z_max - front_z_min
        if side_height < front_height * 0.5:
            # Side view height is suspiciously small, use front
            print(f"  Warning: Side view height ({side_height}) seems too small, using front view height ({front_height})")
            global_z_min = front_z_min
            global_z_max = front_z_max
        else:
            print(f"  Using side view for Z: z_min={global_z_min}, z_max={global_z_max}, height={global_z_max - global_z_min}")
    elif front_z_max:
        print(f"  Using front view for Z: z_min={global_z_min}, z_max={global_z_max}, height={global_z_max - global_z_min}")
    
    if global_z_min is None or global_z_max is None:
        raise ValueError("Could not extract Z dimensions from side or front view")
    
    # Detect step heights from interior lines
    # Horizontal lines in front/side indicate different Z levels
    front_lines = front_data.get("lines", [])
    side_lines = side_data.get("lines", [])
    
    z_levels = [global_z_min, global_z_max]  # Start with full height
    
    # Extract horizontal lines (constant Z) from front view
    for line in front_lines:
        start = line.get("start", [0, 0])
        end = line.get("end", [0, 0])
        if abs(start[1] - end[1]) < 2:  # Horizontal line
            z_level = (start[1] + end[1]) / 2
            if global_z_min < z_level < global_z_max:
                z_levels.append(z_level)
    
    # Extract horizontal lines from side view
    for line in side_lines:
        start = line.get("start", [0, 0])
        end = line.get("end", [0, 0])
        if abs(start[1] - end[1]) < 2:  # Horizontal line
            z_level = (start[1] + end[1]) / 2
            if global_z_min < z_level < global_z_max:
                z_levels.append(z_level)
    
    z_levels = sorted(set(z_levels))
    
    # Assign Z ranges to regions
    # For now, assign full height to all regions
    # Later, we can use LLM to determine which regions are tall/short
    for region in regions:
        region.z_min = global_z_min
        region.z_max = global_z_max
    
    return regions


def solve_geometry(
    top_path: str,
    side_path: str,
    front_path: str
) -> Dict[str, Any]:
    """
    Main geometry solver: extracts all numeric dimensions using priority system.
    
    Returns:
        {
            "regions": [
                {
                    "id": "region_0",
                    "x": 14.0,
                    "y": 6.0,
                    "width": 87.0,
                    "depth": 59.0,
                    "z_min": 0.0,
                    "z_max": 87.0
                },
                ...
            ],
            "metadata": {
                "x_source": "top_view",
                "y_source": "top_view",
                "z_source": "front_view"  # or "side_view"
            }
        }
    """
    top_data = load_json(top_path)
    side_data = load_json(side_path)
    front_data = load_json(front_path)
    
    # Step 1: Extract X, Y from top view (priority 1)
    x_min, y_min, width, depth = extract_x_y_from_top(top_data)
    print(f"  Top view footprint: x={x_min}, y={y_min}, width={width}, depth={depth}")
    
    # Step 2: Split into regions based on interior lines
    top_lines = top_data.get("lines", [])
    print(f"  Found {len(top_lines)} interior lines in top view")
    regions = split_regions_by_lines(x_min, y_min, width, depth, top_lines)
    print(f"  Created {len(regions)} regions from splits")
    
    # Step 3: Extract Z from side/front (priority: Side > Front)
    regions = extract_z_from_side_front(side_data, front_data, regions)
    
    # Determine which view was used for Z
    side_shapes = side_data.get("shapes", [])
    z_source = "side_view" if side_shapes else "front_view"
    
    # Convert to JSON-serializable format
    result = {
        "regions": [
            {
                "id": r.id,
                "x": r.x,
                "y": r.y,
                "width": r.width,
                "depth": r.depth,
                "z_min": r.z_min,
                "z_max": r.z_max,
                "height": r.z_max - r.z_min
            }
            for r in regions
        ],
        "metadata": {
            "x_source": "top_view",
            "y_source": "top_view",
            "z_source": z_source,
            "total_regions": len(regions)
        }
    }
    
    return result


if __name__ == "__main__":
    # Test the geometry solver
    import argparse
    
    parser = argparse.ArgumentParser(description="Extract geometry from 2D views using priority system")
    parser.add_argument("--top", default="2D Json/top_view.json", help="Top view JSON")
    parser.add_argument("--side", default="2D Json/side_view.json", help="Side view JSON")
    parser.add_argument("--front", default="2D Json/front_view.json", help="Front view JSON")
    parser.add_argument("--output", default="geometry_regions.json", help="Output regions JSON")
    
    args = parser.parse_args()
    
    result = solve_geometry(args.top, args.side, args.front)
    
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    
    print(f"✓ Extracted {len(result['regions'])} regions")
    print(f"✓ Saved to {args.output}")
