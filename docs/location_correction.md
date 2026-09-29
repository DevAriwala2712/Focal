# Wayanad event-location correction

The task brief supplied approximately **11.782°N, 76.233°E** as the Wayanad landslide centre. Phase 0 used that point exactly, and its SCL audit remains a valid record for the supplied AOI. It does **not** cover the July 2024 Chooralmala–Mundakkai landslide site.

[NRSC/ISRO's 31 July 2024 Chooralmala impact map](https://bhuvan-app1.nrsc.gov.in/disaster/usrtasks/landslide/doc/Charter_1029_VAP_3_31july2024.pdf) marks the landslide location at **76°8′10.58″E, 11°28′0.347″N** (decimal **76.136272°E, 11.466763°N**). The WGS84 geodesic distance from the brief's point is **36.43 km**. These numbers come from the map's printed coordinates and `pyproj.Geod(ellps='WGS84')`, not from a guessed scene. The original 10 km square cannot intersect the mapped site.

The `configs/pipeline.yaml` event AOI now uses NRSC's point. `risk/results/event_aoi/` contains a separate annual STAC/SCL audit; Phase 0's original evidence is retained unchanged. The Streamlit viewer checks the source AOI and refuses to display assets made for the supplied but incorrect point. A successful corrected-site demo still only indicates vegetation disturbance unless supported by independent temporal labels.
