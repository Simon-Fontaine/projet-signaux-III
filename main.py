import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, measure, transform, morphology
from skimage.transform import resize
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import distance
import glob
import os
import warnings

warnings.filterwarnings("ignore")

MIN_AREA = 800
ASPECT_RATIO_MIN = 1.2
CROP_PADDING = 0.15

REF_COLORS_RGB = {
    "black": (20, 20, 20),
    "brown": (100, 50, 30),
    "red": (200, 40, 40),
    "orange": (230, 110, 20),
    "yellow": (240, 220, 30),
    "green": (40, 160, 40),
    "blue": (40, 60, 190),
    "violet": (120, 50, 180),
    "gray": (128, 128, 128),
    "white": (240, 240, 240),
    "gold": (140, 110, 40),
    "silver": (180, 180, 190),
}

RESISTOR_VALUES = {
    "black": (0, 1, None),
    "brown": (1, 10, 1),
    "red": (2, 100, 2),
    "orange": (3, 1000, None),
    "yellow": (4, 10000, None),
    "green": (5, 100000, 0.5),
    "blue": (6, 1000000, 0.25),
    "violet": (7, 10000000, 0.1),
    "gray": (8, None, 0.05),
    "white": (9, None, None),
    "gold": (None, 0.1, 5),
    "silver": (None, 0.01, 10),
}


def get_ref_lab_colors():
    ref_lab = {}
    for name, rgb in REF_COLORS_RGB.items():
        rgb_norm = np.array([[rgb]], dtype=float) / 255.0
        lab = color.rgb2lab(rgb_norm)[0][0]
        ref_lab[name] = lab
    return ref_lab


REF_COLORS_LAB = get_ref_lab_colors()


def load_and_preprocess(image_path):
    try:
        img = io.imread(image_path)
        if len(img.shape) > 2 and img.shape[2] == 4:
            img = img[:, :, :3]
        return filters.median(img, footprint=np.ones((3, 3, 1)))
    except Exception as e:
        print(f"Erreur critique lors du chargement : {e}")
        return None


def segment_and_crop(image):
    scale_factor = 0.25
    h_orig, w_orig = image.shape[:2]
    img_small = np.array(
        resize(
            image,
            (int(h_orig * scale_factor), int(w_orig * scale_factor)),
            anti_aliasing=True,
            preserve_range=True,
        )
    ).astype(np.uint8)

    hsv = color.rgb2hsv(img_small)
    sat = hsv[:, :, 1]
    try:
        thresh_s = filters.threshold_otsu(sat)
        mask_s = sat > thresh_s
    except:
        mask_s = np.zeros(img_small.shape[:2], dtype=bool)

    gray = color.rgb2gray(img_small)
    try:
        thresh_g = filters.threshold_otsu(gray)
        mask_g = gray < thresh_g
        if np.sum(mask_g) > mask_g.size / 0.6:
            mask_g = ~mask_g
    except:
        mask_g = np.zeros(img_small.shape[:2], dtype=bool)

    binary = np.logical_or(mask_s, mask_g)

    binary_clean = morphology.binary_erosion(binary, morphology.disk(2))
    binary_clean = morphology.binary_opening(binary_clean, morphology.disk(2))
    binary_clean = morphology.binary_dilation(binary_clean, morphology.disk(4))

    label_image = measure.label(binary_clean)
    regions = measure.regionprops(label_image)

    resistor = None
    max_area = 0
    min_area_scaled = MIN_AREA * (scale_factor**2)

    for r in regions:
        if r.area < min_area_scaled:
            continue
        if r.axis_minor_length == 0:
            continue
        if r.axis_major_length / r.axis_minor_length < ASPECT_RATIO_MIN:
            continue

        if r.area > max_area:
            max_area = r.area
            resistor = r

    if not resistor:
        return None

    angle = -np.degrees(resistor.orientation)

    centroid_orig = (
        resistor.centroid[0] / scale_factor,
        resistor.centroid[1] / scale_factor,
    )

    edge_pixels = np.concatenate(
        [image[0, :, :], image[-1, :, :], image[:, 0, :], image[:, -1, :]]
    )
    bg_color = np.median(edge_pixels, axis=0)

    rotated_img = np.array(
        transform.rotate(
            image,
            angle,
            center=centroid_orig,
            resize=True,
            preserve_range=True,
            order=1,
            cval=0,
        )
    ).astype(np.uint8)

    mask_valid = np.ones(image.shape[:2], dtype=bool)
    rotated_mask_valid = transform.rotate(
        mask_valid, angle, center=centroid_orig, resize=True, order=0
    ).astype(bool)
    for c in range(3):
        channel = rotated_img[:, :, c]
        channel[~rotated_mask_valid] = bg_color[c]
        rotated_img[:, :, c] = channel

    binary_orig_size = np.array(
        resize(binary_clean, (h_orig, w_orig), order=0, anti_aliasing=False)
    )
    rotated_mask = transform.rotate(
        binary_orig_size, angle, center=centroid_orig, resize=True, order=0
    )

    rows = np.any(rotated_mask, axis=1)
    cols = np.any(rotated_mask, axis=0)
    if not np.any(rows) or not np.any(cols):
        return None

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    roi_mask = rotated_mask[ymin:ymax, xmin:xmax]
    col_heights = np.sum(roi_mask, axis=0)

    if len(col_heights) > 0:
        median_height = np.median(col_heights[col_heights > 0])
        valid_cols = np.where(col_heights > median_height * 0.7)[0]

        if len(valid_cols) > 0:
            new_xmin = xmin + valid_cols[0]
            new_xmax = xmin + valid_cols[-1]

            pad_x = int((new_xmax - new_xmin) * 0.05)
            xmin = max(0, new_xmin - pad_x)
            xmax = min(rotated_img.shape[1], new_xmax + pad_x)

            h_box = ymax - ymin
            pad_y = int(h_box * CROP_PADDING)
            ymin = max(0, ymin - pad_y)
            ymax = min(rotated_img.shape[0], ymax + pad_y)

    cropped = rotated_img[ymin:ymax, xmin:xmax]

    if cropped.shape[0] > cropped.shape[1]:
        cropped = transform.rotate(
            cropped, 90, resize=True, preserve_range=True
        ).astype(np.uint8)

    return cropped


def extract_signals(crop):
    h, w, _ = crop.shape
    roi = crop[int(h * 0.45) : int(h * 0.55), :]

    line_rgb = np.median(roi, axis=0).reshape(1, w, 3).astype(np.uint8)
    line_lab = color.rgb2lab(line_rgb).reshape(w, 3)
    return line_rgb.reshape(w, 3), line_lab


def detect_peaks_robust(line_lab):
    bg_lab = np.median(line_lab, axis=0)
    delta_e = np.sqrt(np.sum((line_lab - bg_lab) ** 2, axis=1))

    sigma = max(2, len(delta_e) * 0.015)
    delta_e_smooth = gaussian_filter1d(delta_e, sigma=sigma)

    min_dist = len(delta_e) / 15
    height_thresh = np.max(delta_e_smooth) * 0.25

    peaks, props = find_peaks(
        delta_e_smooth,
        height=height_thresh,
        distance=min_dist,
        prominence=5,
    )
    return peaks, props, delta_e, delta_e_smooth


def classify_color_knn(sample_rgb):
    sample_lab = color.rgb2lab(np.array([[sample_rgb]], dtype=float) / 255.0)[0][0]
    sample_hsv = color.rgb2hsv(np.array([[sample_rgb]], dtype=float) / 255.0)[0][0]
    s_val = sample_hsv[1]
    v_val = sample_hsv[2]

    min_dist = float("inf")
    best_color = "unknown"

    for name, ref_lab in REF_COLORS_LAB.items():
        d = distance.euclidean(sample_lab, ref_lab)
        penalty = 0

        if v_val < 0.3 and name in ["white", "yellow", "silver"]:
            penalty = 50

        if s_val < 0.15 and name not in ["black", "white", "gray", "silver"]:
            penalty = 50

        final_score = d + penalty

        if final_score < min_dist:
            min_dist = final_score
            best_color = name

    return best_color


def decode_resistor(colors):
    if len(colors) < 3:
        return "Indéterminé"
    bands = list(colors)

    if bands[0] in ["gold", "silver"] and bands[-1] not in ["gold", "silver"]:
        bands.reverse()

    try:
        if len(bands) >= 3:
            vals = [RESISTOR_VALUES[b][0] for b in bands if b in RESISTOR_VALUES]
            mults = [RESISTOR_VALUES[b][1] for b in bands if b in RESISTOR_VALUES]
            tols = [RESISTOR_VALUES[b][2] for b in bands if b in RESISTOR_VALUES]

            tol_val = 20
            if tols[-1] is not None:
                tol_val = tols[-1]
                vals = vals[:-1]
                mults = mults[:-1]

            ohm_val = 0

            if len(vals) >= 2:
                mult_val = mults[-1]
                digits = vals[:-1]

                if mult_val is None:
                    return "Erreur Multiplicateur"

                val_acc = 0
                for d in digits:
                    if d is not None:
                        val_acc = val_acc * 10 + d

                ohm_val = val_acc * mult_val
            else:
                return "Erreur Structure"

            suffix = "Ω"
            if ohm_val >= 1e6:
                ohm_val /= 1e6
                suffix = "MΩ"
            elif ohm_val >= 1e3:
                ohm_val /= 1e3
                suffix = "kΩ"

            return f"{ohm_val:.2f} {suffix} +/- {tol_val}%"

    except Exception as e:
        pass

    return "Erreur Décodage"


def show_visu(crop, peaks, colors, result, delta_e_smooth):
    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.imshow(crop)
    plt.title(f"Résultat: {result}", color="blue", fontweight="bold")
    plt.axis("off")

    for i, p in enumerate(peaks):
        plt.axvline(x=p, color="red", linestyle="--")
        txt = colors[i] if i < len(colors) else "?"
        plt.text(
            p, 0, txt, color="yellow", fontsize=9, ha="center", backgroundcolor="black"
        )

    plt.subplot(2, 1, 2)
    plt.plot(delta_e_smooth, color="purple", label="Variation Couleur (Delta E)")
    plt.plot(peaks, delta_e_smooth[peaks], "x", color="red", label="Pics détectés")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    img_dir = "./images"
    files = glob.glob(os.path.join(img_dir, "*.[jJ][pP][gG]")) + glob.glob(
        os.path.join(img_dir, "*.[pP][nN][gG]")
    )

    print(f"Traitement de {len(files)} images...")

    for f in files:
        if "Figure" in f:
            continue

        print(f"\n--- Analyse de {os.path.basename(f)} ---")

        img = load_and_preprocess(f)
        if img is None:
            continue

        crop = segment_and_crop(img)
        if crop is None:
            print(">> Echec : Impossible d'isoler la résistance.")
            continue

        rgb_line, lab_line = extract_signals(crop)
        peaks, _, _, delta_smooth = detect_peaks_robust(lab_line)

        colors = []
        for p in peaks:
            s = max(0, p - 2)
            e = min(len(rgb_line), p + 3)
            sample = np.mean(rgb_line[s:e], axis=0)
            colors.append(classify_color_knn(sample))

        val_str = decode_resistor(colors)
        print(f">> Bandes détectées : {colors}")
        print(f">> Résultat final   : {val_str}")

        show_visu(crop, peaks, colors, val_str, delta_smooth)


if __name__ == "__main__":
    main()
