import argparse
import json
from dataclasses import dataclass, asdict
from typing import List, Literal, Tuple, Dict, Any

import cv2
import numpy as np


ViewType = Literal["front", "top", "side"]
ShapeType = Literal["rectangle", "circle", "polygon"]


@dataclass
class RectangleShape:
    id: str
    type: ShapeType
    view: ViewType
    params: Dict[str, float]
    vertices: List[Tuple[float, float]]


@dataclass
class CircleShape:
    id: str
    type: ShapeType
    view: ViewType
    center: Tuple[float, float]
    radius: float


@dataclass
class PolygonShape:
    id: str
    type: ShapeType
    view: ViewType
    vertices: List[Tuple[float, float]]


@dataclass
class LineShape:
    id: str
    type: str
    view: ViewType
    start: Tuple[float, float]
    end: Tuple[float, float]


def load_image(path: str) -> np.ndarray:
    image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Could not read image at path: {path}")
    return image


def preprocess(image: np.ndarray) -> np.ndarray:
    # Basic denoising and binarization.
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Otsu thresholding assumes dark lines on light background.
    _, thresh = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    return thresh


def find_contours(bin_img: np.ndarray) -> List[np.ndarray]:
    contours, _ = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    return contours


def _is_border_line_for_rect(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    rect: RectangleShape,
    coord_tol: float = 3.0,
    coverage_ratio: float = 0.6,
) -> bool:
    """
    Heuristic: treat a line as a border if it lies very close to one of the
    rectangle's sides and covers most of that side's length.
    """
    rx = rect.params["x"]
    ry = rect.params["y"]
    rw = rect.params["width"]
    rh = rect.params["height"]

    # Normalize so we can reason about ranges.
    lx_min = min(x1, x2)
    lx_max = max(x1, x2)
    ly_min = min(y1, y2)
    ly_max = max(y1, y2)

    # Horizontal candidate: y nearly constant.
    if abs(y1 - y2) < coord_tol:
        # Near top edge?
        if abs(y1 - ry) < coord_tol:
            overlap = min(lx_max, rx + rw) - max(lx_min, rx)
            if overlap > coverage_ratio * rw:
                return True
        # Near bottom edge?
        if abs(y1 - (ry + rh)) < coord_tol:
            overlap = min(lx_max, rx + rw) - max(lx_min, rx)
            if overlap > coverage_ratio * rw:
                return True

    # Vertical candidate: x nearly constant.
    if abs(x1 - x2) < coord_tol:
        # Near left edge?
        if abs(x1 - rx) < coord_tol:
            overlap = min(ly_max, ry + rh) - max(ly_min, ry)
            if overlap > coverage_ratio * rh:
                return True
        # Near right edge?
        if abs(x1 - (rx + rw)) < coord_tol:
            overlap = min(ly_max, ry + rh) - max(ly_min, ry)
            if overlap > coverage_ratio * rh:
                return True

    return False


def _is_border_line(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    rectangles: List[RectangleShape],
) -> bool:
    """
    Decide whether a given line segment is just outlining an existing
    rectangle (outer frame or inner rectangular feature border).
    """
    for rect in rectangles:
        if _is_border_line_for_rect(x1, y1, x2, y2, rect):
            return True
    return False


def detect_lines(
    bin_img: np.ndarray,
    view: ViewType,
    img_shape: Tuple[int, int],
    rectangles: List[RectangleShape],
    min_length_ratio: float = 0.1,
) -> List[LineShape]:
    """
    Detect straight line segments using probabilistic Hough transform.
    Returns line segments as LineShape objects.
    """
    # Use Canny on the same binary image to get clean edges.
    edges = cv2.Canny(bin_img, 50, 150, apertureSize=3)

    h, w = img_shape
    min_length = min(h, w) * min_length_ratio

    raw_lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=int(min_length),
        maxLineGap=5,
    )

    if raw_lines is None:
        return []

    lines: List[LineShape] = []
    for idx, l in enumerate(raw_lines):
        x1, y1, x2, y2 = [float(v) for v in l[0]]

        # Skip very short segments (extra safety).
        if np.hypot(x2 - x1, y2 - y1) < min_length:
            continue

        # Skip lines that simply coincide with rectangle borders.
        if _is_border_line(x1, y1, x2, y2, rectangles):
            continue

        # Create line object; Stage 2 will decide if it's a split/edge/etc.
        lines.append(
            LineShape(
                id=f"line_{idx}",
                type="line",
                view=view,
                start=(float(x1), float(y1)),
                end=(float(x2), float(y2)),
            )
        )

    return lines


def contour_to_shape(
    contour: np.ndarray,
    view: ViewType,
    idx: int,
    eps_factor: float = 0.02,
) -> Any:
    # Polygon approximation
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, eps_factor * peri, True)

    # Ignore tiny noise
    area = cv2.contourArea(contour)
    if area < 10.0:
        return None

    # Rectangle: 4 vertices, nearly axis-aligned
    if len(approx) == 4:
        pts = approx.reshape(-1, 2).astype(float)
        x, y, w, h = cv2.boundingRect(pts.astype(np.int32))

        # Check axis-alignment by seeing if edges are mostly horizontal/vertical
        # Compute edge directions
        diffs = np.diff(np.vstack([pts, pts[0]]), axis=0)
        angles = np.degrees(np.arctan2(diffs[:, 1], diffs[:, 0]))
        angles = np.mod(angles, 180.0)
        # Horizontal ~0 or 180, vertical ~90
        aligned_edges = sum(
            (abs(a - 0) < 10)
            or (abs(a - 90) < 10)
            or (abs(a - 180) < 10)
            for a in angles
        )
        if aligned_edges >= 3:
            return RectangleShape(
                id=f"rect_{idx}",
                type="rectangle",
                view=view,
                params={
                    "x": float(x),
                    "y": float(y),
                    "width": float(w),
                    "height": float(h),
                },
                vertices=[(float(px), float(py)) for px, py in pts],
            )

    # Circle: many vertices and close to a circle
    if len(approx) >= 5:
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        cx, cy, radius = float(cx), float(cy), float(radius)
        circle_area = np.pi * radius * radius
        if circle_area <= 0:
            return None
        # If the contour area is close to the circle area, treat it as circle.
        ratio = area / circle_area
        if 0.7 < ratio < 1.3:
            return CircleShape(
                id=f"circle_{idx}",
                type="circle",
                view=view,
                center=(cx, cy),
                radius=radius,
            )

    # Fallback: generic polygon
    pts = approx.reshape(-1, 2).astype(float)
    return PolygonShape(
        id=f"poly_{idx}",
        type="polygon",
        view=view,
        vertices=[(float(px), float(py)) for px, py in pts],
    )


def extract_shapes(image_path: str, view: ViewType) -> Dict[str, Any]:
    image = load_image(image_path)
    bin_img = preprocess(image)
    contours = find_contours(bin_img)

    shapes: List[Any] = []
    for idx, contour in enumerate(contours):
        shape = contour_to_shape(contour, view=view, idx=idx)
        if shape is not None:
            shapes.append(shape)

    rectangles: List[RectangleShape] = [
        s for s in shapes if isinstance(s, RectangleShape)
    ]

    lines = detect_lines(
        bin_img,
        view=view,
        img_shape=image.shape,
        rectangles=rectangles,
    )

    return {
        "view": view,
        "image_path": image_path,
        "shapes": [asdict(s) for s in shapes],
        "lines": [asdict(l) for l in lines],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract 2D geometric shapes (rectangles, circles, polygons) "
            "from a PNG orthographic view and export them as JSON."
        )
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the PNG image containing the drawing.",
    )
    parser.add_argument(
        "--view",
        type=str,
        choices=["front", "top", "side"],
        required=True,
        help="Which orthographic view this image represents.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to output JSON file.",
    )

    args = parser.parse_args()

    data = extract_shapes(args.image_path, args.view)  # type: ignore[arg-type]
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Extracted {len(data['shapes'])} shapes from {args.image_path}")
    print(f"Saved JSON to {args.output}")


if __name__ == "__main__":
    main()

