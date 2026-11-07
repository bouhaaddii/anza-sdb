# NASA HLS scene: 10 November 2022

- Product: `HLS.S30.T29RMP.2022314T112209.v2.0`
- Sensor product: Harmonized Landsat and Sentinel-2, Sentinel-2 MSI (S30)
- CMR acquisition interval: 2022-11-10 11:32:47–11:32:56 UTC
- MGRS tile: T29RMP
- CMR cloud-cover estimate: 2% (catalog granule-level estimate)
- Pixel size: 30 m for all HLS S30 bands
- Input bands used: B02 (blue), B03 (green), B04 (red), and B08 (broad NIR)
- Quality layer: Fmask; bits 1, 2, 3, and 4 are used to exclude cloud, cloud/shadow adjacency, cloud shadow, and snow/ice; fill value 255 is excluded. The water bit is retained as a QA diagnostic rather than a hard training filter.
- Reflectance scale: stored integer values are multiplied by 0.0001; fill value is -9999.
- Data source: NASA LP DAAC HLS S30 v2.0, DOI [10.5067/HLS/HLSS30.002](https://doi.org/10.5067/HLS/HLSS30.002).
- Product guide: [HLS User Guide v2.0](https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf).

The date is confirmed by the user as the topobathymetric survey date. The product's catalog timestamp is not a measurement timestamp for each individual survey point. The exact survey start/end times and water-level/tidal conditions are not documented in the supplied survey plan. HLS S30 provides 30 m pixels; it cannot preserve the original Sentinel-2 10 m spatial detail. Model training therefore aggregates survey labels by 30 m HLS pixel and must not treat repeated 10 m survey-grid points within a single HLS pixel as independent image observations.
