"""Build one sample per HLS 30 m pixel from the Anza survey and scene."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol, xy
from rasterio.warp import transform

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SCENE = RAW / "sentinel2" / "HLS_S30_2022-11-10"
PROCESSED = ROOT / "data" / "processed"
SURVEY_CRS = "EPSG:26192"
BANDS = ("B02", "B03", "B04", "B08")
PRODUCT = "HLS.S30.T29RMP.2022314T112209.v2.0"
PATCH = 9
RADIUS = PATCH // 2
REFLECTANCE_SCALE = 0.0001
QA_BAD_BITS = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)


def load_rasters():
    paths = {band: SCENE / f"{PRODUCT}.{band}.tif" for band in BANDS}
    paths["Fmask"] = SCENE / f"{PRODUCT}.Fmask.tif"
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing downloaded HLS files:\n" + "\n".join(missing))

    opened = {name: rasterio.open(path) for name, path in paths.items()}
    reference = opened["B02"]
    for name, src in opened.items():
        if (src.crs != reference.crs or src.transform != reference.transform
                or src.width != reference.width or src.height != reference.height):
            raise ValueError(f"{name} is not on the B02 grid.")
    arrays = {name: src.read(1) for name, src in opened.items()}
    meta = {
        "crs": reference.crs.to_string(),
        "transform": tuple(reference.transform),
        "width": reference.width,
        "height": reference.height,
        "pixel_size_m": float(abs(reference.transform.a)),
        "reflectance_scale": REFLECTANCE_SCALE,
        "fill_value": int(reference.nodata),
    }
    for src in opened.values():
        src.close()
    return arrays, reference, meta


def make_dataset() -> dict:
    arrays, reference, raster_meta = load_rasters()
    points = np.genfromtxt(RAW / "survey" / "bathy_points.txt", delimiter="\t", dtype=str)
    point_ids = points[:, 0]
    sx = points[:, 1].astype("float64")
    sy = points[:, 2].astype("float64")
    z = points[:, 3].astype("float64")
    x, y = transform(SURVEY_CRS, reference.crs, sx.tolist(), sy.tolist())
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    rows, cols = rowcol(reference.transform, x, y)
    rows = np.asarray(rows, dtype="int32")
    cols = np.asarray(cols, dtype="int32")
    h, w = arrays["Fmask"].shape
    in_grid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    indices = np.flatnonzero(in_grid)
    row_inside, col_inside = rows[indices], cols[indices]
    cell_keys = row_inside.astype("int64") * w + col_inside
    unique_keys, inverse = np.unique(cell_keys, return_inverse=True)
    n_total = np.bincount(inverse)
    n_negative = np.bincount(inverse, weights=(z[indices] < 0).astype("int32"))
    n_positive = n_total - n_negative
    negative_sum = np.bincount(inverse, weights=np.where(z[indices] < 0, z[indices], 0.0))
    z_mean = np.divide(negative_sum, n_negative, out=np.zeros_like(negative_sum), where=n_negative > 0)

    ur = (unique_keys // w).astype("int32")
    uc = (unique_keys % w).astype("int32")
    qa = arrays["Fmask"][ur, uc]
    qa_clear = (qa != 255) & ((qa & QA_BAD_BITS) == 0)
    spectral_clear = np.ones(len(unique_keys), dtype=bool)
    for band in BANDS:
        spectral_clear &= arrays[band][ur, uc] != -9999
    # Keep cells with a sufficiently sampled underwater surface and no mixed land/water survey labels.
    eligible = (n_negative >= 3) & (n_positive == 0) & qa_clear & spectral_clear
    target_grid = np.zeros((h, w), dtype="float32")
    target_mask = np.zeros((h, w), dtype=bool)
    target_grid[ur[eligible], uc[eligible]] = z_mean[eligible].astype("float32")
    target_mask[ur[eligible], uc[eligible]] = True
    cell_keys = unique_keys[eligible]
    rr, cc = ur[eligible], uc[eligible]
    labels = z_mean[eligible].astype("float32")
    counts = n_negative[eligible].astype("int32")
    qa_kept = qa[eligible].astype("uint8")
    if not len(cell_keys):
        raise ValueError("No eligible HLS pixels remain after survey and Fmask quality filtering.")

    # Use a principal coastal axis, then leave a patch-width buffer between spatial holdout zones.
    px, py = xy(reference.transform, rr, cc, offset="center")
    xy_centres = np.column_stack([px, py]).astype("float64")
    centre = xy_centres.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(xy_centres - centre, full_matrices=False)
    alongshore_axis = vt[0]
    projected = (xy_centres - centre) @ alongshore_axis
    q1, q2 = np.quantile(projected, [1 / 3, 2 / 3])
    buffer_m = RADIUS * raster_meta["pixel_size_m"] * np.sqrt(2)
    zones = np.full(len(cell_keys), -1, dtype="int8")
    zones[projected < q1 - buffer_m] = 0
    zones[(projected > q1 + buffer_m) & (projected < q2 - buffer_m)] = 1
    zones[projected > q2 + buffer_m] = 2

    # Form radiometrically scaled 9x9 patches. All patch pixels must be valid.
    offsets = np.arange(-RADIUS, RADIUS + 1)
    keep_window = (
        (rr >= RADIUS) & (rr < h - RADIUS) &
        (cc >= RADIUS) & (cc < w - RADIUS) &
        (zones >= 0)
    )
    candidates = np.flatnonzero(keep_window)
    X_list, y_list, id_list, r_list, c_list, zone_list, count_list, qa_list = [], [], [], [], [], [], [], []
    bad_qa_patch_count = 0
    fill_patch_count = 0
    for q in candidates:
        r, c = int(rr[q]), int(cc[q])
        slices = (slice(r - RADIUS, r + RADIUS + 1), slice(c - RADIUS, c + RADIUS + 1))
        patch = np.stack([arrays[b][slices] for b in BANDS], axis=-1)
        qa_patch = arrays["Fmask"][slices]
        patch_has_bad_qa = np.any((qa_patch == 255) | ((qa_patch & QA_BAD_BITS) != 0))
        patch_has_fill = np.any(patch == -9999)
        bad_qa_patch_count += int(patch_has_bad_qa)
        fill_patch_count += int(patch_has_fill)
        if patch_has_fill or patch_has_bad_qa:
            continue
        X_list.append(patch.astype("float32") * REFLECTANCE_SCALE)
        y_list.append(labels[q])
        id_list.append(f"r{r:04d}_c{c:04d}")
        r_list.append(r)
        c_list.append(c)
        zone_list.append(int(zones[q]))
        count_list.append(int(counts[q]))
        qa_list.append(int(qa_kept[q]))
    if not X_list:
        raise ValueError("No HLS patches with complete reflectance support remain.")
    X = np.stack(X_list).astype("float32")
    y_values = np.asarray(y_list, dtype="float32")
    row_keep = np.asarray(r_list, dtype="int32")
    col_keep = np.asarray(c_list, dtype="int32")
    zone_keep = np.asarray(zone_list, dtype="int8")
    point_counts = np.asarray(count_list, dtype="int32")
    qa_values = np.asarray(qa_list, dtype="uint8")
    point_qa = arrays["Fmask"][rows[indices], cols[indices]]

    ratio = np.log(
        np.maximum(X[:, RADIUS, RADIUS, 0], 1e-5)
        / np.maximum(X[:, RADIUS, RADIUS, 1], 1e-5)
    ).astype("float32")[:, None]
    target_patches = np.zeros((len(X), PATCH, PATCH, 1), dtype="float32")
    target_weights = np.zeros_like(target_patches)
    for k, (r, c) in enumerate(zip(row_keep, col_keep)):
        sl = (slice(r - RADIUS, r + RADIUS + 1), slice(c - RADIUS, c + RADIUS + 1))
        target_patches[k, :, :, 0] = target_grid[sl]
        target_weights[k, :, :, 0] = target_mask[sl].astype("float32")
    x_coords, y_coords = xy(reference.transform, row_keep, col_keep, offset="center")
    x_coords = np.asarray(x_coords, dtype="float64")
    y_coords = np.asarray(y_coords, dtype="float64")
    pixel_ids = np.asarray(id_list, dtype=str)

    output = PROCESSED / "samples.npz"
    np.savez_compressed(
        output, X=X, y=y_values[:, None], ratio=ratio,
        rows=row_keep, cols=col_keep, zones=zone_keep,
        target_patches=target_patches, target_weights=target_weights,
        point_ids=pixel_ids, pixel_ids=pixel_ids,
        x_utm=x_coords, y_utm=y_coords,
        survey_points_per_pixel=point_counts, fmask_values=qa_values,
        target_grid=target_grid, target_grid_mask=target_mask,
    )
    zone_counts = {str(i): int((zone_keep == i).sum()) for i in range(3)}
    summary = {
        "site": "Anza, Agadir, Morocco",
        "product_id": PRODUCT,
        "acquisition_date": "2022-11-10",
        "acquisition_interval_utc": "11:32:47-11:32:56",
        "sensor_product": "NASA HLS S30, Sentinel-2 MSI",
        "tile": "T29RMP",
        "catalog_cloud_cover_percent": 2,
        "spatial_resolution_m": raster_meta["pixel_size_m"],
        "bands": list(BANDS),
        "reflectance_conversion": "stored int16 reflectance multiplied by 0.0001; fill value -9999",
        "image_crs": raster_meta["crs"],
        "survey_source_crs_from_plan": SURVEY_CRS,
        "survey_date": "2022-11-10 (confirmed by the user)",
        "survey_z_unit": "m (confirmed by the user)",
        "survey_time": "not documented",
        "vertical_reference": "Hydrographic zero of the Port of Agadir, Morocco (confirmed by the survey-data owner)",
        "vertical_reference_status": "confirmed by the survey-data owner; no levelling tie to the Port of Ras Kebdana datum is documented",
        "quality_mask": "exclude fill=255, cloud bit 1, adjacent cloud/shadow bit 2, shadow bit 3, snow/ice bit 4; retain water bit 5 as a diagnostic",
        "training_unit": "one observation per 30 m HLS pixel; negative survey elevations averaged within pixel",
        "minimum_survey_points_per_pixel": 3,
        "mixed_sign_cells": "excluded when the same HLS pixel contains both negative and non-negative survey points",
        "patch_shape": list(X.shape[1:]),
        "patch_radius_m": float(RADIUS * raster_meta["pixel_size_m"]),
        "sample_count_unique_hls_pixel_centres": int(len(y_values)),
        "candidate_centres_inside_spatial_zones": int(len(candidates)),
        "candidate_windows_with_bad_fmask_qa": int(bad_qa_patch_count),
        "candidate_windows_with_reflectance_fill": int(fill_patch_count),
        "zone_counts": zone_counts,
        "survey_point_count": int(len(z)),
        "survey_points_in_hls_tile": int(in_grid.sum()),
        "negative_survey_points_in_tile": int(((z < 0) & in_grid).sum()),
        "negative_survey_points_on_clear_fmask_pixels": int((((z[indices] < 0) & ((point_qa & QA_BAD_BITS) == 0) & (point_qa != 255)).sum())),
        "unique_pixels_with_negative_survey_points": int((n_negative > 0).sum()),
        "negative_pixels_with_fmask_cloud_shadow_or_snow_flags": int(((n_negative > 0) & (~qa_clear)).sum()),
        "negative_pixels_with_hls_fill_in_any_input_band": int(((n_negative > 0) & (~spectral_clear)).sum()),
        "negative_pixels_with_fewer_than_three_survey_points": int(((n_negative > 0) & (n_negative < 3)).sum()),
        "mixed_sign_survey_pixels_excluded": int(((n_negative > 0) & (n_positive > 0)).sum()),
        "eligible_pixels_before_spatial_split": int(eligible.sum()),
        "eligible_reference_grid_cells": int(target_mask.sum()),
        "mean_survey_points_per_retained_pixel": float(point_counts.mean()),
        "retained_pixels_with_hls_water_bit": int(((qa_values & (1 << 5)) != 0).sum()),
        "spatial_split": "three alongshore zones from the first principal coordinate axis; zones separated by at least the projected 9x9 patch footprint; boundaries rotate through test/validation/train roles",
        "pca_axis_xy": [float(v) for v in alongshore_axis],
        "pca_variance_ratio_axis1": float((singular_values[0] ** 2) / np.sum(singular_values ** 2)),
        "source_label_caution": "Survey date, z unit (metres), and reference to the hydrographic zero of the Port of Agadir were confirmed by the survey-data owner. Survey time, measurement uncertainty, and tide/water-level correction are not documented. Metrics describe errors against aggregated labels in the Anza local datum. Ras Kebdana uses a separate local port datum, and no vertical tie between the two port zeros is documented.",
        "sample_file": "data/processed/samples.npz",
    }
    (PROCESSED / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    print(json.dumps(make_dataset(), indent=2))


if __name__ == "__main__":
    main()
