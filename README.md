## 2D → 3D Parametric Engine

Turn three orthographic 2D views (front, top, side) into a 3D mesh using OpenCV, Gemini, and trimesh.

---

### 1. Overview

Pipeline:

1. `extract_2d_shapes.py`  
   PNG/JPEG view → 2D JSON (`rectangles`, `circles`, `polygons`, `lines`).
2. `reconstruct_3d_features.py`  
   2D JSON + images → 3D feature JSON (via Gemini).
3. `house_generator.py`  
   3D feature JSON → 3D mesh (`.stl`, `.obj`, etc.).

Key tech:

- **OpenCV** – 2D geometry extraction from images
- **Gemini** – infers 3D features from multiple views
- **trimesh + shapely + numpy** – 3D solid modeling & export

Repo structure (important parts):

- `extract_2d_shapes.py` – 2D feature extractor
- `reconstruct_3d_features.py` – Gemini-based 3D reconstruction
- `house_generator.py` – JSON → mesh generator CLI
- `req.txt` – Python dependencies
- `2D Json/` – 2D view JSONs (`front_view.json`, `top_view.json`, `side_view.json`)
- `Ref Image/` – raw images for each orthographic view
- `model_features.json`, `model_features_refined.json` – sample 3D feature JSONs

---

### 2. Prerequisites

- Windows 10+
- Python 3.10.x (project venv is 3.10.0)
- A Gemini API key with sufficient quota

You can use the existing virtual environment in `3D/` or your own.

---

### 3. Setup

From the project root: `D:\Projects\2d To 3D`

#### 3.1 Activate venv and install deps

```powershell
cd "D:\Projects\2d To 3D"
.\3D\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r req.txt
```

This installs:

- `trimesh`, `shapely`, `numpy`, `pyglet`
- `opencv-python`
- `google-generativeai`
- `python-dotenv`
- `pillow`

#### 3.2 Configure Gemini API key

Create `.env` in the project root:

```env
GEMINI_API_KEY=YOUR_REAL_API_KEY_HERE
```

`reconstruct_3d_features.py` will load this automatically via `python-dotenv`.

---

### 4. End-to-End Workflow

#### Step 1 – Extract 2D geometry (OpenCV)

Run once per orthographic view (filenames can be changed as needed):

```powershell
# From project root, venv active
python extract_2d_shapes.py "Ref Image\Front.jpeg" --view front --output "2D Json\front_view.json"
python extract_2d_shapes.py "Ref Image\Top.jpeg"   --view top   --output "2D Json\top_view.json"
python extract_2d_shapes.py "Ref Image\Side.jpeg"  --view side  --output "2D Json\side_view.json"
```

Each `*_view.json` contains:

- `view`: `"front" | "top" | "side"`
- `image_path`: image filename (used later)
- `shapes`: rectangles/circles/polygons with `params` and `vertices`
- `lines`: internal line segments (splits, steps, etc.)

Make sure `image_path` matches actual image filenames (e.g. `front.jpeg` in `Ref Image/`).

#### Step 2 – Reconstruct 3D features (Gemini)

`reconstruct_3d_features.py`:

- Loads `2D Json/front_view.json`, `top_view.json`, `side_view.json`
- Loads referenced images (via `image_path`)
- Sends **images + JSON + detailed instructions** to Gemini
- Writes `model_features.json` (3D feature JSON)

Run:

```powershell
python reconstruct_3d_features.py `
  --front "2D Json\front_view.json" `
  --top   "2D Json\top_view.json"   `
  --side  "2D Json\side_view.json"  `
  --output "model_features.json"    `
  --model "gemini-flash-latest"
```

Notes:

- Requires a valid Gemini key with quota; otherwise you’ll see 429 / quota errors.
- If images can’t be loaded (bad `image_path`), it falls back to text-only mode (less accurate).

#### Step 3 – JSON → 3D mesh (trimesh)

`house_generator.py` is a CLI that:

- Loads a feature JSON (`model_features.json`)
- Supports:
  - Old format: each feature has `depth`
  - New format: each feature has `dimensions.depth`
- Performs extrudes and cuts to build a solid
- Exports to STL / OBJ / PLY / OFF / DAE / GLB

Basic usage:

```powershell
python house_generator.py model_features.json
```

This:

- Loads `model_features.json`
- Generates a mesh
- Exports `model_features.stl` by default

Custom output and format:

```powershell
python house_generator.py model_features.json -o table.obj --format obj
```

Supported `--format` values:

- `stl`, `obj`, `ply`, `off`, `dae`, `glb`

You can then open the mesh in MeshLab, Blender, or a CAD tool.

---

### 5. Typical Workflow for a New Part

1. Place orthographic images in `Ref Image/`:
   - e.g. `Front.jpeg`, `Top.jpeg`, `Side.jpeg`
2. Run `extract_2d_shapes.py` for each view to populate `2D Json/`.
3. Sanity-check the JSON (e.g. `front_view.json`) to confirm shapes/lines look reasonable.
4. Run `reconstruct_3d_features.py` → produce `model_features.json`.
5. Run `house_generator.py model_features.json` → generate `model_features.stl` (or other format).
6. Inspect the 3D model in your viewer/CAD and iterate as needed.

---

### 6. Troubleshooting

- **Gemini 429 / quota errors**  
  - Message: `ResourceExhausted: 429 ... quota exceeded`  
  - Fix: Check/project billing & quota in Google Cloud / AI Studio, or switch to another project/model.

- **`GEMINI_API_KEY` not set**  
  - Ensure `.env` exists with `GEMINI_API_KEY=...` in the project root.
  - Restart your shell/IDE or confirm current working directory is the project root.

- **Images not found**  
  - Confirm `image_path` in each `*_view.json` is correct (e.g. `front.jpeg`).  
  - Ensure those files exist (commonly in `Ref Image/`).  

- **Empty or invalid mesh**  
  - Check that `model_features.json` has a non-empty `"features"` array.  
  - Each feature must include `profile`, `type`, and either `depth` or `dimensions.depth`.  

- **Boolean (union/difference) warnings**  
  - Sometimes geometric booleans fail on degenerate shapes; warnings are printed and the script continues.

---

### 7. Editing / Extending

- **2D detection tuning** – `extract_2d_shapes.py`  
  - Change thresholding, contour filtering, rectangle/circle classification, and border-line filtering here.

- **3D schema + Gemini behavior** – `reconstruct_3d_features.py`  
  - `build_prompt`: controls instructions and desired JSON schema.  
  - Can adjust coordinate system assumptions, feature schema, and extraction rules.

- **Mesh generation / feature engine** – `house_generator.py`  
  - `create_extruded_mesh`: how rectangle/circle/polygon profiles become meshes.  
  - `generate_model`: main feature loop (extrude vs cut, union vs difference).  
  - CLI `main()`: JSON loading and file export options.

---

### 8. How to Push to GitHub

From the project root (with git already initialized and remote set):

```powershell
cd "D:\Projects\2d To 3D"
git status
git add README.md
git commit -m "Add README with 2D-to-3D pipeline instructions"
git push origin main   # or your branch name
```

Replace `main` with your actual default branch if different.

