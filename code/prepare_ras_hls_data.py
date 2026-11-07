"""Prepare site-specific Ras El Ma labels and 30 m HLS patches."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol, xy
from rasterio.warp import transform

ROOT = Path(__file__).resolve().parents[1]
POINTS = ROOT / "data" / "raw" / "survey" / "ras_bathy_points.csv"
SCENE = ROOT / "data" / "raw" / "sentinel2" / "HLS_S30_2024-08-07_RasElMa" / "T30SWD"
PRODUCT = "HLS.S30.T30SWD.2024220T105031.v2.0"
PROCESSED = ROOT / "data" / "processed" / "ras_el_ma_2024-08-08"
SURVEY_CRS = "EPSG:26191"  # Merchich / Nord Maroc; Lambert Zone 1 in the survey cartouche.
BANDS = ("B02", "B03", "B04", "B08")
PATCH = 9
RADIUS = PATCH // 2
REFLECTANCE_SCALE = 0.0001
# HLS v2.0 QA: bits 1-4 are cloud, adjacent cloud/shadow, shadow and snow/ice.
# Bit 0 is reserved; water (bit 5) is retained as a diagnostic for shallow-water work.
QA_BAD_BITS = (1 << 1) | (1 << 2) | (1 << 3) | (1 << 4)


def load_rasters():
    paths = {band: SCENE / f"{PRODUCT}.{band}.tif" for band in BANDS}
    paths["Fmask"] = SCENE / f"{PRODUCT}.Fmask.tif"
    missing = [str(path) for path in paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing HLS files:\n" + "\n".join(missing))

    opened = {name: rasterio.open(path) for name, path in paths.items()}
    try:
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
    finally:
        for src in opened.values():
            src.close()
    return arrays, reference.transform, meta


def make_dataset() -> dict:
    arrays, affine, raster_meta = load_rasters()
    points = pd.read_csv(POINTS)
    raw_point_count = int(len(points))
    required = {"dxf_handle", "x", "y", "z_source"}
    if not required.issubset(points.columns):
        raise ValueError(f"Survey CSV must contain {sorted(required)}")
    if not np.isfinite(points[["x", "y", "z_source"]].to_numpy(dtype="float64")).all():
        raise ValueError("Survey CSV contains non-finite x, y, or z values.")
    unique_xyz = points.drop_duplicates(subset=["x", "y", "z_source"])
    duplicate_exact_xyz_rows = raw_point_count - int(len(unique_xyz))
    xy_stats = unique_xyz.groupby(["x", "y"])["z_source"].agg(["min", "max", "size"]).reset_index()
    xy_z_spread = xy_stats["max"] - xy_stats["min"]
    xy_points = unique_xyz.groupby(["x", "y"], as_index=False).agg(z_source=("z_source", "median"))
    sx = xy_points["x"].to_numpy(dtype="float64")
    sy = xy_points["y"].to_numpy(dtype="float64")
    z = xy_points["z_source"].to_numpy(dtype="float64")

    x, y = transform(SURVEY_CRS, raster_meta["crs"], sx.tolist(), sy.tolist())
    x = np.asarray(x, dtype="float64")
    y = np.asarray(y, dtype="float64")
    rows, cols = rowcol(affine, x, y)
    rows = np.asarray(rows, dtype="int32")
    cols = np.asarray(cols, dtype="int32")
    h, w = arrays["Fmask"].shape
    in_grid = (rows >= 0) & (rows < h) & (cols >= 0) & (cols < w)
    indices = np.flatnonzero(in_grid)
    if not len(indices):
        raise ValueError("No survey points fall inside the selected HLS granule.")

    cell_keys = rows[indices].astype("int64") * w + cols[indices]
    unique_keys, inverse = np.unique(cell_keys, return_inverse=True)
    n_total = np.bincount(inverse)
    n_negative = np.bincount(inverse, weights=(z[indices] < 0).astype("int32"))
    n_nonnegative = n_total - n_negative
    negative_sum = np.bincount(inverse, weights=np.where(z[indices] < 0, z[indices], 0.0))
    z_mean = np.divide(negative_sum, n_negative, out=np.zeros_like(negative_sum), where=n_negative > 0)

    ur = (unique_keys // w).astype("int32")
    uc = (unique_keys % w).astype("int32")
    qa = arrays["Fmask"][ur, uc]
    qa_clear = (qa != 255) & ((qa & QA_BAD_BITS) == 0)
    spectral_clear = np.ones(len(unique_keys), dtype=bool)
    for band in BANDS:
        spectral_clear &= arrays[band][ur, uc] != -9999
    eligible = (n_negative >= 3) & (n_nonnegative == 0) & qa_clear & spectral_clear

    target_grid = np.zeros((h, w), dtype="float32")
    target_mask = np.zeros((h, w), dtype=bool)
    target_grid[ur[eligible], uc[eligible]] = z_mean[eligible].astype("float32")
    target_mask[ur[eligible], uc[eligible]] = True
    rr, cc = ur[eligible], uc[eligible]
    labels = z_mean[eligible].astype("float32")
    counts = n_negative[eligible].astype("int32")
    qa_kept = qa[eligible].astype("uint8")
    if not len(labels):
        raise ValueError("No Ras El Ma HLS pixels remain after survey and Fmask filtering.")

    # Use a principal coastal axis and keep a 9 x 9 patch footprint out of fold boundaries.
    px, py = xy(affine, rr, cc, offset="center")
    xy_centres = np.column_stack([px, py]).astype("float64")
    centre = xy_centres.mean(axis=0)
    _, singular_values, vt = np.linalg.svd(xy_centres - centre, full_matrices=False)
    alongshore_axis = vt[0]
    projected = (xy_centres - centre) @ alongshore_axis
    q1, q2 = np.quantile(projected, [1 / 3, 2 / 3])
    buffer_m = RADIUS * raster_meta["pixel_size_m"] * np.sqrt(2)
    zones = np.full(len(labels), -1, dtype="int8")
    zones[projected < q1 - buffer_m] = 0
    zones[(projected > q1 + buffer_m) & (projected < q2 - buffer_m)] = 1
    zones[projected > q2 + buffer_m] = 2

    offsets = np.arange(-RADIUS, RADIUS + 1)
    in_patch = ((rr >= RADIUS) & (rr < h - RADIUS) &
                (cc >= RADIUS) & (cc < w - RADIUS) & (zones >= 0))
    candidates = np.flatnonzero(in_patch)
    X_list, y_list, ids, r_list, c_list, zone_list, count_list, qa_list = [], [], [], [], [], [], [], []
    bad_qa_patch_count = fill_patch_count = 0
    for q in candidates:
        r, c = int(rr[q]), int(cc[q])
        slices = (slice(r - RADIUS, r + RADIUS + 1), slice(c - RADIUS, c + RADIUS + 1))
        patch = np.stack([arrays[b][slices] for b in BANDS], axis=-1)
        qa_patch = arrays["Fmask"][slices]
        bad_qa = np.any((qa_patch == 255) | ((qa_patch & QA_BAD_BITS) != 0))
        has_fill = np.any(patch == -9999)
        bad_qa_patch_count += int(bad_qa)
        fill_patch_count += int(has_fill)
        if bad_qa or has_fill:
            continue
        X_list.append(patch.astype("float32") * REFLECTANCE_SCALE)
        y_list.append(labels[q])
        ids.append(f"r{r:04d}_c{c:04d}")
        r_list.append(r)
        c_list.append(c)
        zone_list.append(int(zones[q]))
        count_list.append(int(counts[q]))
        qa_list.append(int(qa_kept[q]))
    if not X_list:
        raise ValueError("No Ras El Ma 9 x 9 patches have complete reflectance support.")

    X = np.stack(X_list).astype("float32")
    y_values = np.asarray(y_list, dtype="float32")
    row_keep = np.asarray(r_list, dtype="int32")
    col_keep = np.asarray(c_list, dtype="int32")
    zone_keep = np.asarray(zone_list, dtype="int8")
    point_counts = np.asarray(count_list, dtype="int32")
    qa_values = np.asarray(qa_list, dtype="uint8")
    ratio = np.log(np.maximum(X[:, RADIUS, RADIUS, 0], 1e-5) /
                   np.maximum(X[:, RADIUS, RADIUS, 1], 1e-5)).astype("float32")[:, None]
    target_patches = np.zeros((len(X), PATCH, PATCH, 1), dtype="float32")
    target_weights = np.zeros_like(target_patches)
    for k, (r, c) in enumerate(zip(row_keep, col_keep)):
        sl = (slice(r - RADIUS, r + RADIUS + 1), slice(c - RADIUS, c + RADIUS + 1))
        target_patches[k, :, :, 0] = target_grid[sl]
        target_weights[k, :, :, 0] = target_mask[sl].astype("float32")
    x_coords, y_coords = xy(affine, row_keep, col_keep, offset="center")
    output = PROCESSED / "samples.npz"
    np.savez_compressed(
        output, X=X, y=y_values[:, None], ratio=ratio,
        rows=row_keep, cols=col_keep, zones=zone_keep,
        target_patches=target_patches, target_weights=target_weights,
        point_ids=np.asarray(ids, dtype=str), pixel_ids=np.asarray(ids, dtype=str),
        x_utm=np.asarray(x_coords, dtype="float64"), y_utm=np.asarray(y_coords, dtype="float64"),
        survey_points_per_pixel=point_counts, fmask_values=qa_values,
        target_grid=target_grid, target_grid_mask=target_mask,
    )
    zone_counts = {str(i): int((zone_keep == i).sum()) for i in range(3)}
    summary = {
        "site": "Ras El Ma / Ras Kebdana, Morocco",
        "product_id": PRODUCT,
        "imagery_acquisition_date": "2024-08-07",
        "survey_date": "2024-08-08",
        "survey_z_unit": "m",
        "vertical_reference": "Local hydrographic zero of the Port of Ras Kebdana",
        "horizontal_crs_used": SURVEY_CRS,
        "horizontal_crs_basis": "The plan states Lambert Zone 1, Clarke 1880, Lambert Conic; the GPKG metadata identifies EPSG:26191 (Merchich / Nord Maroc). The DXF-derived points align spatially with the HLS T30SWD footprint after this transformation.",
        "datum_translation_parameters_in_plan_m": {"dx": -31, "dy": -146, "dz": -47},
        "horizontal_translation_caution": "The plan lists geodetic datum translations, but its source and target datum realizations and transformation convention are not fully documented. They are recorded here and were not applied as an additional manual shift. The ΔZ parameter is not a tie between local hydrographic zeros.",
        "hls_collection": "NASA/USGS HLS S30 v2.0",
        "hls_granule_cloud_cover_percent": 3,
        "hls_resolution_m": raster_meta["pixel_size_m"],
        "survey_to_image_day_difference": -1,
        "image_crs": raster_meta["crs"],
        "bands": list(BANDS),
        "reflectance_conversion": "stored int16 reflectance multiplied by 0.0001; fill value -9999",
        "quality_mask": "Exclude fill=255, cloud bit 1, adjacent cloud/shadow bit 2, shadow bit 3 and snow/ice bit 4; HLS v2.0 bit 0 is reserved and water bit 5 is retained as a diagnostic.",
        "raw_survey_rows": raw_point_count,
        "duplicate_exact_xyz_rows_removed": duplicate_exact_xyz_rows,
        "unique_xyz_rows_after_deduplication": int(len(unique_xyz)),
        "unique_xy_coordinates_after_vertical_median": int(len(xy_points)),
        "repeated_xy_groups_with_multiple_z_before_collapse": int((xy_stats["size"] > 1).sum()),
        "xy_groups_with_conflicting_z": int((xy_z_spread > 1e-9).sum()),
        "maximum_z_spread_at_repeated_xy_m": float(xy_z_spread.max()) if len(xy_z_spread) else 0.0,
        "unique_xy_coordinates_inside_selected_hls_granule": int(in_grid.sum()),
        "survey_coordinate_coverage_fraction": float(in_grid.mean()),
        "unique_xy_coordinates_with_negative_z": int((z < 0).sum()),
        "unique_pixels_with_negative_survey_points": int((n_negative > 0).sum()),
        "negative_pixels_with_bad_fmask_flags": int(((n_negative > 0) & (~qa_clear)).sum()),
        "negative_pixels_with_fill_in_an_input_band": int(((n_negative > 0) & (~spectral_clear)).sum()),
        "negative_pixels_with_fewer_than_three_points": int(((n_negative > 0) & (n_negative < 3)).sum()),
        "mixed_sign_pixels_excluded": int(((n_negative > 0) & (n_nonnegative > 0)).sum()),
        "eligible_pixels_before_spatial_split": int(eligible.sum()),
        "candidate_centres_inside_spatial_zones": int(len(candidates)),
        "candidate_windows_with_bad_fmask_qa": int(bad_qa_patch_count),
        "candidate_windows_with_reflectance_fill": int(fill_patch_count),
        "sample_count_unique_hls_pixel_centres": int(len(X)),
        "mean_survey_points_per_retained_pixel": float(point_counts.mean()),
        "retained_pixels_with_hls_water_bit": int(((qa_values & (1 << 5)) != 0).sum()),
        "zone_counts": zone_counts,
        "patch_shape": list(X.shape[1:]),
        "patch_radius_m": float(RADIUS * raster_meta["pixel_size_m"]),
        "spatial_split": "Three alongshore zones from the first principal coordinate axis; one patch-radius buffer between zones; fold roles rotate across zones.",
        "pca_variance_ratio_axis1": float(singular_values[0] ** 2 / np.sum(singular_values ** 2)),
        "source_label_caution": "Exact duplicate xyz rows are removed; repeated xy coordinates are represented by median z, with the maximum observed spread recorded above. Metrics are relative to aggregated negative survey z values in the Port of Ras Kebdana local hydrographic datum. The HLS image is one day before the survey. No water-level correction, point uncertainty, or vertical tie to the Port of Agadir datum is available.",
        "sample_file": "data/processed/ras_el_ma_2024-08-08/samples.npz",
    }
    (PROCESSED / "dataset_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    print(json.dumps(make_dataset(), indent=2))


if __name__ == "__main__":
    main()

