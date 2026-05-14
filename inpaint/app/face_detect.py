"""Face detection module using OpenCV Haar cascades.

Provides face region detection and automatic mask generation
for face reconstruction tasks.
"""
import io
import numpy as np

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    HAS_OPENCV = False


# Standard face region proportions (relative to face bounding box)
FACE_REGIONS = {
    'left_eye':  {'x': 0.12, 'y': 0.22, 'w': 0.35, 'h': 0.22},
    'right_eye': {'x': 0.53, 'y': 0.22, 'w': 0.35, 'h': 0.22},
    'nose':      {'x': 0.28, 'y': 0.38, 'w': 0.44, 'h': 0.30},
    'mouth':     {'x': 0.22, 'y': 0.62, 'w': 0.56, 'h': 0.22},
    'forehead':  {'x': 0.12, 'y': 0.00, 'w': 0.76, 'h': 0.25},
    'left_cheek': {'x': 0.00, 'y': 0.35, 'w': 0.28, 'h': 0.35},
    'right_cheek': {'x': 0.72, 'y': 0.35, 'w': 0.28, 'h': 0.35},
    'chin':      {'x': 0.22, 'y': 0.80, 'w': 0.56, 'h': 0.20},
}


def detect_faces(image_array):
    """Detect faces in an image using OpenCV Haar cascade.

    Parameters
    ----------
    image_array : np.ndarray
        RGB image array of shape (H, W, 3)

    Returns
    -------
    faces : list of dict
        Each dict has keys: x, y, w, h (as fractions of image size),
        and 'regions' with sub-regions for face parts.
    """
    if not HAS_OPENCV:
        # Fallback: assume whole image is a face (works for CelebA)
        return [{
            'x': 0.1, 'y': 0.05, 'w': 0.8, 'h': 0.9,
            'regions': _compute_regions(0.1, 0.05, 0.8, 0.9)
        }]

    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    h, w = gray.shape[:2]

    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    faces_raw = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(int(w * 0.15), int(h * 0.15))
    )

    if len(faces_raw) == 0:
        # No face detected — assume centered face (CelebA default)
        return [{
            'x': 0.1, 'y': 0.05, 'w': 0.8, 'h': 0.9,
            'regions': _compute_regions(0.1, 0.05, 0.8, 0.9)
        }]

    faces = []
    for (fx, fy, fw, fh) in faces_raw:
        # Normalize to [0, 1]
        nx, ny = fx / w, fy / h
        nw, nh = fw / w, fh / h
        faces.append({
            'x': float(nx), 'y': float(ny),
            'w': float(nw), 'h': float(nh),
            'regions': _compute_regions(nx, ny, nw, nh)
        })

    return faces


def _compute_regions(face_x, face_y, face_w, face_h):
    """Compute face sub-regions from face bounding box (all in [0,1] coords)."""
    regions = {}
    for name, props in FACE_REGIONS.items():
        rx = face_x + props['x'] * face_w
        ry = face_y + props['y'] * face_h
        rw = props['w'] * face_w
        rh = props['h'] * face_h
        regions[name] = {
            'x': round(float(rx), 4),
            'y': round(float(ry), 4),
            'w': round(float(rw), 4),
            'h': round(float(rh), 4)
        }
    return regions


def generate_region_mask(image_size, regions_to_mask, face_data):
    """Generate a binary mask for specified face regions.

    Parameters
    ----------
    image_size : tuple
        (width, height) of the image
    regions_to_mask : list of str
        Region names from FACE_REGIONS
    face_data : dict
        Face detection result with 'regions' key

    Returns
    -------
    mask : np.ndarray
        Binary mask of shape (height, width) with 1=keep, 0=inpaint
    """
    w, h = image_size
    mask = np.ones((h, w), dtype=np.float32)

    for region_name in regions_to_mask:
        if region_name in face_data.get('regions', {}):
            r = face_data['regions'][region_name]
            x1 = int(r['x'] * w)
            y1 = int(r['y'] * h)
            x2 = int((r['x'] + r['w']) * w)
            y2 = int((r['y'] + r['h']) * h)
            # Create elliptical mask for more natural look
            cy, cx = (y1 + y2) // 2, (x1 + x2) // 2
            ry, rx = (y2 - y1) // 2, (x2 - x1) // 2
            Y, X = np.ogrid[:h, :w]
            ellipse = ((X - cx) / max(rx, 1)) ** 2 + ((Y - cy) / max(ry, 1)) ** 2
            mask[ellipse <= 1.0] = 0.0

    return mask
