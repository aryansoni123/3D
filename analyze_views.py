"""
Diagnostic script to analyze 2D view JSONs and show what Gemini should be seeing.
"""
import json
from pathlib import Path

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def analyze_view(view_name, data):
    print(f"\n{'='*60}")
    print(f"ANALYZING {view_name.upper()} VIEW")
    print(f"{'='*60}")
    
    shapes = data.get("shapes", [])
    lines = data.get("lines", [])
    
    print(f"\nShapes found: {len(shapes)}")
    for shape in shapes:
        print(f"  - {shape['id']}: {shape['type']}")
        if shape['type'] == 'rectangle':
            params = shape.get('params', {})
            print(f"    Position: x={params.get('x')}, y={params.get('y')}")
            print(f"    Size: width={params.get('width')}, height={params.get('height')}")
            print(f"    Vertices: {shape.get('vertices', [])}")
        elif shape['type'] == 'circle':
            print(f"    Center: {shape.get('center')}, Radius: {shape.get('radius')}")
    
    print(f"\nInterior lines found: {len(lines)}")
    for line in lines:
        print(f"  - {line['id']}: from {line['start']} to {line['end']}")
    
    return shapes, lines

def cross_reference_views(front, top, side):
    print(f"\n{'='*60}")
    print("CROSS-REFERENCE ANALYSIS")
    print(f"{'='*60}")
    
    # Front view: X-Z plane
    front_rect = front['shapes'][0] if front['shapes'] else None
    if front_rect:
        fx = front_rect['params']['x']
        fw = front_rect['params']['width']
        fz_bottom = front_rect['params']['y']  # bottom in image = low Z
        fh = front_rect['params']['height']
        fz_top = fz_bottom + fh
        
        print(f"\nFront View (X-Z plane):")
        print(f"  X range: {fx} to {fx + fw} (width = {fw})")
        print(f"  Z range: {fz_bottom} to {fz_top} (height = {fh})")
    
    # Top view: X-Y plane
    top_rect = top['shapes'][0] if top['shapes'] else None
    if top_rect:
        tx = top_rect['params']['x']
        tw = top_rect['params']['width']
        ty_back = top_rect['params']['y']  # back in image = low Y
        td = top_rect['params']['height']  # depth in top view
        ty_front = ty_back + td
        
        print(f"\nTop View (X-Y plane):")
        print(f"  X range: {tx} to {tx + tw} (width = {tw})")
        print(f"  Y range: {ty_back} to {ty_front} (depth = {td})")
    
    # Side view: Y-Z plane
    side_rect = side['shapes'][0] if side['shapes'] else None
    if side_rect:
        sy = side_rect['params']['x']  # Y in side view is horizontal
        sd = side_rect['params']['width']  # depth in side view
        sz_bottom = side_rect['params']['y']  # Z bottom
        sh = side_rect['params']['height']  # height
        sz_top = sz_bottom + sh
        
        print(f"\nSide View (Y-Z plane):")
        print(f"  Y range: {sy} to {sy + sd} (depth = {sd})")
        print(f"  Z range: {sz_bottom} to {sz_top} (height = {sh})")
    
    # Check for alignment
    print(f"\n{'='*60}")
    print("ALIGNMENT CHECK")
    print(f"{'='*60}")
    
    if front_rect and top_rect:
        x_match = abs(fx - tx) < 5  # Allow small tolerance
        x_width_match = abs(fw - tw) < 5
        print(f"X-axis alignment: {'✓' if x_match else '✗'} (front x={fx}, top x={tx})")
        print(f"X-axis width match: {'✓' if x_width_match else '✗'} (front w={fw}, top w={tw})")
    
    if front_rect and side_rect:
        z_match = abs(fz_bottom - sz_bottom) < 5
        z_height_match = abs(fh - sh) < 5
        print(f"Z-axis alignment: {'✓' if z_match else '✗'} (front z={fz_bottom}, side z={sz_bottom})")
        print(f"Z-axis height match: {'✓' if z_height_match else '✗'} (front h={fh}, side h={sh})")
    
    if top_rect and side_rect:
        y_match = abs(ty_back - sy) < 5
        y_depth_match = abs(td - sd) < 5
        print(f"Y-axis alignment: {'✓' if y_match else '✗'} (top y={ty_back}, side y={sy})")
        print(f"Y-axis depth match: {'✓' if y_depth_match else '✗'} (top d={td}, side d={sd})")
    
    # Interior lines analysis
    print(f"\n{'='*60}")
    print("INTERIOR LINES ANALYSIS")
    print(f"{'='*60}")
    
    front_lines = [l for l in front.get('lines', []) if abs(l['start'][0] - l['end'][0]) < 2]  # vertical lines
    top_lines = [l for l in top.get('lines', []) if abs(l['start'][1] - l['end'][1]) < 2]  # horizontal lines in top
    
    print(f"Front view vertical lines (potential X splits): {len(front_lines)}")
    for line in front_lines:
        x_pos = (line['start'][0] + line['end'][0]) / 2
        print(f"  - Line at X ≈ {x_pos}")
    
    print(f"Top view horizontal lines (potential Y splits): {len(top_lines)}")
    for line in top_lines:
        y_pos = (line['start'][1] + line['end'][1]) / 2
        print(f"  - Line at Y ≈ {y_pos}")

if __name__ == "__main__":
    front = load_json("2D Json/front_view.json")
    top = load_json("2D Json/top_view.json")
    side = load_json("2D Json/side_view.json")
    
    analyze_view("FRONT", front)
    analyze_view("TOP", top)
    analyze_view("SIDE", side)
    
    cross_reference_views(front, top, side)
    
    print(f"\n{'='*60}")
    print("RECOMMENDED 3D FEATURE")
    print(f"{'='*60}")
    print("""
Based on the analysis above, the 3D feature should have:
- position.x = minimum X from front/top views
- position.y = minimum Y from top/side views  
- position.z = minimum Z from front/side views
- dimensions.width = X span (should match front and top)
- dimensions.depth = Y span (should match top and side)
- dimensions.height = Z span (should match front and side)
    """)
