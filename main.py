import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, filters, measure, transform, morphology
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
        print(f"Erreur lors du chargement de l'image : {e}")
        return None


def segment_and_crop(image):
    hsv = color.rgb2hsv(image)
    sat = hsv[:, :, 1]
    try:
        thresh_s = filters.threshold_otsu(sat)
        mask_s = sat > thresh_s
    except:
        mask_s = np.zeros(image.shape[:2], dtype=bool)

    gray = color.rgb2gray(image)
    try:
        thresh_g = filters.threshold_otsu(gray)
        mask_g = gray < thresh_g
        if np.sum(mask_g) > mask_g.size / 2:
            mask_g = gray > thresh_g
    except:
        return None

    binary = np.logical_or(mask_s, mask_g)
    binary_clean = morphology.binary_opening(binary, morphology.disk(5))

    if np.sum(binary_clean) < MIN_AREA:
        binary_clean = binary

    binary_clean = morphology.binary_closing(binary_clean, morphology.disk(5))

    label_image = measure.label(binary_clean)
    regions = measure.regionprops(label_image)

    resistor = None
    max_area = 0

    for r in regions:
        if r.area < MIN_AREA:
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

    rotated_img = transform.rotate(
        image,
        angle,
        center=resistor.centroid,
        resize=True,
        preserve_range=True,
        order=1,
    ).astype(np.uint8)

    if rotated_img is None or rotated_img.size == 0 or len(rotated_img.shape) < 2:
        return None

    rotated_mask = transform.rotate(
        label_image == resistor.label,
        angle,
        center=resistor.centroid,
        resize=True,
        order=0,
    )

    rows = np.any(rotated_mask, axis=1)
    cols = np.any(rotated_mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return None

    ymin, ymax = np.where(rows)[0][[0, -1]]
    xmin, xmax = np.where(cols)[0][[0, -1]]

    h_box = ymax - ymin
    w_box = xmax - xmin
    pad_y = int(h_box * CROP_PADDING)
    pad_x = int(w_box * CROP_PADDING * 0.5)

    ymin = max(0, ymin - pad_y)
    ymax = min(rotated_img.shape[0], ymax + pad_y)
    xmin = max(0, xmin - pad_x)
    xmax = min(rotated_img.shape[1], xmax + pad_x)

    cropped = rotated_img[ymin:ymax, xmin:xmax]

    if cropped.shape[0] > cropped.shape[1]:
        cropped = transform.rotate(
            cropped, 90, resize=True, preserve_range=True
        ).astype(np.uint8)

    return cropped


def extract_signals(crop):
    h, w, _ = crop.shape
    roi = crop[int(h * 0.4) : int(h * 0.6), :]
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
        if len(bands) >= 4:
            vals = [RESISTOR_VALUES[b][0] for b in bands if b in RESISTOR_VALUES]
            mults = [RESISTOR_VALUES[b][1] for b in bands if b in RESISTOR_VALUES]
            tols = [RESISTOR_VALUES[b][2] for b in bands if b in RESISTOR_VALUES]

            tol_val = 20
            if tols[-1] is not None:
                tol_val = tols[-1]

            mult_val = 1
            if mults[-2] is not None:
                mult_val = mults[-2]

            digits = vals[:-2]
            ohm_val = 0

            if len(digits) == 2 and None not in digits:
                ohm_val = (digits[0] * 10 + digits[1]) * mult_val

            elif len(digits) == 3 and None not in digits:
                ohm_val = (digits[0] * 100 + digits[1] * 10 + digits[2]) * mult_val

            else:
                if len(digits) >= 2 and None not in digits[:2]:
                    ohm_val = (digits[0] * 10 + digits[1]) * mult_val
                else:
                    return "Erreur Valeurs"

            suffix = "Ω"
            if ohm_val >= 1e6:
                ohm_val /= 1e6
                suffix = "MΩ"
            elif ohm_val >= 1e3:
                ohm_val /= 1e3
                suffix = "kΩ"

            return f"{ohm_val:.2f} {suffix} +/- {tol_val}%"

    except Exception as e:
        print(f"Erreur de calcul: {e}")
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
    plt.plot(peaks, delta_e_smooth[peaks], "x", color="red", label="Pics (Bandes)")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def main():
    img_dir = "./images"
    files = glob.glob(os.path.join(img_dir, "*.[jJ][pP][gG]"))

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
            print(
                ">> Echec : Impossible d'isoler la résistance (objet trop petit ou fond complexe)."
            )
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
