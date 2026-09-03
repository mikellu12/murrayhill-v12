# What a clean repository would carry

384 files tracked today. Proposed: keep 240, drop 144.

Nothing is deleted from this machine -- the drop list simply does not go into the new repository.

## Keep


### (root)

- **.gitignore** (8 KB)
- **README.md** (34 KB)  
  Murray Hill streetscape pipeline
- **config.yaml** (34 KB)  
  seed: 42
- **config_london.yaml** (34 KB)  
  seed: 42

### data

- **data/london/nodes_source.csv** (137 KB)
- **data/london/nodes_source.gpkg** (404 KB)
- **data/london/processed/frame_labels.csv** (3 KB)
- **data/london/processed/indoor_batch.csv** (5 KB)
- **data/london/processed/indoor_flag.csv** (523 KB)
- **data/london/processed/indoor_labels.csv** (1 KB)
- **data/london/processed/manifest.csv** (679 KB)
- **data/london/processed/nodes.csv** (404 KB)
- **data/london/processed/nodes.gpkg** (636 KB)
- **data/london/processed/offstreet_flag.csv** (501 KB)
- **data/london/processed/provenance.csv** (42 KB)
- **data/london/processed/seg90_two_model.csv** (6630 KB)
- **data/london/processed/street_type.csv** (109 KB)
- **data/processed/azimuth_profiles.npz** (3847 KB)
- **data/processed/block_faces.csv** (2 KB)
- **data/processed/directional_metrics.csv** (590 KB)
- **data/processed/directional_metrics_wide.csv** (195 KB)
- **data/processed/dob_shed_by_node.csv** (43 KB)
- **data/processed/manifest.csv** (264 KB)
- **data/processed/metrics.csv** (347 KB)
- **data/processed/metrics.gpkg** (400 KB)
- **data/processed/nodes.csv** (209 KB)
- **data/processed/nodes.gpkg** (336 KB)
- **data/processed/nodes_infill.csv** (1 KB)
- **data/processed/nodes_infill.gpkg** (96 KB)
- **data/processed/nodes_v13.gpkg** (244 KB)
- **data/processed/nodes_with_faces.csv** (257 KB)
- **data/processed/osm_nodes_source.csv** (70 KB)
- **data/processed/scaffold.csv** (235 KB)
- **data/processed/scaffold_by_node.csv** (21 KB)
- **data/processed/seg180_two_model.csv** (1646 KB)
- **data/processed/seg90_bands.csv** (420 KB)
- **data/processed/seg90_gmi_band.csv** (131 KB)
- **data/processed/seg90_shares.csv** (1473 KB)
- **data/processed/seg90_two_model.csv** (3241 KB)
- **data/processed/sim_index.csv** (199 KB)
- **data/processed/sim_terms_alongstreet.csv** (127 KB)
- **data/processed/street_axis.csv** (25 KB)
- **data/processed/street_type.csv** (47 KB)
- **data/processed/study_area.geojson** (3 KB)
- **data/processed/study_area_config.json** (4 KB)
- **data/processed/subway_entrance_dist.csv** (31 KB)
- **data/raw/building_footprints.geojson** (2055 KB)
- **data/raw/dob_permits.csv** (551 KB)
- **data/raw/frame_source/README.md** (4 KB)  
  Frame source of record
- **data/raw/frame_source/final_coordinates_mapping.csv** (199 KB)
- **data/raw/frame_source/final_nodes_cleaned.gpkg** (488 KB)
- **data/raw/frame_source/final_nodes_output.gpkg** (372 KB)
- **data/raw/frame_source/superseded_nodes_2026-08-18.gpkg** (208 KB)
- **data/raw/metadata.csv** (134 KB)
- **data/raw/truth_bench.csv** (3 KB)
- **data/raw/truth_bus_shelter.csv** (10 KB)
- **data/raw/truth_street_tree.csv** (920 KB)

### docs

- **docs/figures.md** (1 KB)  
  Figures
- **docs/handpicking.md** (6 KB)  
  Handpicking images for the scaffolding ground truth
- **docs/london.md** (6 KB)  
  The London frame: what the imagery is, and what it costs
- **docs/model_selection.md** (12 KB)  
  Model selection

### (root)

- **gridaxis.py** (3 KB)  
  Position along the Manhattan grid's uptown axis, in metres.
- **main.py** (1 KB)  
  Run the Murray Hill streetscape pipeline.
- **make_dashboard.py** (57 KB)  
  Build a self-contained HTML dashboard from the pipeline outputs.
- **preflight.py** (8 KB)  
  Preflight: is this folder ready to run, and what will it cost?
- **requirements-analysis.txt** (0 KB)  
  Analysis only -- no GPU, no torch, no transformers.
- **requirements.txt** (1 KB)  
  Install CUDA torch FIRST -- plain PyPI gives the CPU-only wheel:

### results

- **results/DATA_NOTES.md** (4 KB)  
  Reading these tables
- **results/london/tables/sim_vlm_london.csv** (5001 KB)
- **results/london/tables/vlm_calculations.csv** (3218 KB)
- **results/london/tables/vlm_calculations_london.csv** (3218 KB)
- **results/london/tables/vlm_observations.csv** (5437 KB)
- **results/london/tables/vlm_observations_london.csv** (5437 KB)
- **results/tables/README.md** (13 KB)  
  SIM run tables
- **results/tables/block_matrices.docx** (37 KB)
- **results/tables/block_matrix_counts.csv** (0 KB)
- **results/tables/block_matrix_sim.csv** (0 KB)
- **results/tables/by_street.csv** (2 KB)
- **results/tables/dob_sheds.csv** (554 KB)
- **results/tables/dwell_demo.csv** (58 KB)
- **results/tables/enclosure_bands.csv** (0 KB)
- **results/tables/enclosure_envelope.csv** (1 KB)
- **results/tables/enclosure_envelope_directional.csv** (4 KB)
- **results/tables/enclosure_invertedU.csv** (0 KB)
- **results/tables/enclosure_shape_race.csv** (0 KB)
- **results/tables/face_sample_shares.csv** (6 KB)
- **results/tables/face_samples.csv** (3 KB)
- **results/tables/gwr_machinery.csv** (1 KB)
- **results/tables/model_benchmark.csv** (2 KB)
- **results/tables/nodes_per_block.csv** (5 KB)
- **results/tables/nodes_per_street.csv** (1 KB)
- **results/tables/nodes_without_metrics.csv** (3 KB)
- **results/tables/offtarget_nodes.csv** (2 KB)
- **results/tables/open_one_side_audit.csv** (3 KB)
- **results/tables/pano_overlap.csv** (97 KB)
- **results/tables/partial_correlations.csv** (0 KB)
- **results/tables/rating_logits.csv** (19 KB)
- **results/tables/rating_reliability.csv** (1 KB)
- **results/tables/regression_along_cross.csv** (0 KB)
- **results/tables/regression_by_direction.csv** (1 KB)
- **results/tables/regression_gvi_vei.csv** (1 KB)
- **results/tables/robust_associations.csv** (2 KB)
- **results/tables/s03_subset_diff.csv** (8 KB)
- **results/tables/scale_width.csv** (41 KB)
- **results/tables/segmentation_categories.csv** (2 KB)
- **results/tables/sim_samples.csv** (1 KB)
- **results/tables/sim_vlm.csv** (342 KB)
- **results/tables/sim_vlm_180_placeless.csv** (1180 KB)
- **results/tables/sim_vlm_fix_smoke.csv** (2 KB)
- **results/tables/sim_vlm_ias.csv** (493 KB)
- **results/tables/sim_vlm_smoke.csv** (4 KB)
- **results/tables/sim_vlm_v2.csv** (2392 KB)
- **results/tables/sim_vlm_v2_oldrungs.csv** (1935 KB)
- **results/tables/sim_vlm_v2_smoke.csv** (10 KB)
- **results/tables/sim_vlm_v2_validation.csv** (1 KB)
- **results/tables/sim_vlm_v3.csv** (2388 KB)
- **results/tables/sim_vlm_v4_clean.csv** (2091 KB)
- **results/tables/sim_vlm_validation.csv** (1 KB)
- **results/tables/sim_vlm_with_arcs.csv** (2174 KB)
- **results/tables/svi_180_openvocab.csv** (205 KB)
- **results/tables/svi_180_qwen_morphology.csv** (439 KB)
- **results/tables/svi_180_scaffold.csv** (182 KB)
- **results/tables/svi_180_segformer.csv** (251 KB)
- **results/tables/svi_180_sim_vlm.csv** (473 KB)
- **results/tables/svi_180_visual_labels.csv** (24 KB)
- **results/tables/svi_180_vlm.csv** (117 KB)
- **results/tables/svi_90_sim.csv** (200 KB)
- **results/tables/svi_90_sim.partial_fullschema.csv** (200 KB)
- **results/tables/validation_twins.csv** (1 KB)
- **results/tables/vlm_benchmark.csv** (1 KB)
- **results/tables/vlm_calculations.csv** (1654 KB)
- **results/tables/vlm_observations.csv** (3080 KB)
- **results/tables/vlm_prompts.txt** (9 KB)  
  ==============================================================================
- **results/tables/vlm_rung_ab.csv** (190 KB)
- **results/tables/vlm_sections.csv** (8 KB)

### (root)

- **run_analysis.py** (3 KB)  
  Laptop entry point -- every analysis stage, no GPU required.

### src

- **src/common.py** (16 KB)  
  Shared configuration, paths, keys and geometry helpers.
- **src/mast.py** (8 KB)  
  Find and remove Google's camera mast from a Street View frame.
- **src/s01_frame.py** (8 KB)  
  Stage 1 -- sampling frame.
- **src/s02_imagery.py** (9 KB)  
  Stage 2 -- Street View metadata and imagery.
- **src/s03_profiles.py** (7 KB)  
  Stage 3 -- azimuthal class profiles.
- **src/s04_metrics.py** (13 KB)  
  Stage 4 -- metrics, coverage audit, spatial structure.
- **src/s05_geometry.py** (20 KB)  
  Stage 5 -- building heights, measured street width, H/W.
- **src/s06_analysis.py** (22 KB)  
  Stage 6 -- regression and group contrasts.
- **src/s07_enclosure.py** (29 KB)  
  Stage 7 -- GVI against H/W, on the framework's pre-specified bands.
- **src/s08_figures.py** (10 KB)  
  Stage 7 -- figures.
- **src/sim_core.py** (7 KB)  
  The Street Interface Matrix exactly as the manuscript defines it.
- **src/sim_fields.py** (6 KB)  
  The nine VLM ratings that supply every term in the manuscript's SIM.
- **src/sim_scale.py** (11 KB)  
  Seven verbal anchors per field, not two.

### tools

- **tools/anchor_score.py** (3 KB)  
  Score anchor_probe.csv against the measured share over each 90-degree arc.
- **tools/block_counts.py** (5 KB)  
  Node counts per street and per street block.
- **tools/block_matrix.py** (4 KB)  
  The cross-streets as a matrix: 9 streets by 5 blocks, laid out as the map.
- **tools/block_matrix_docx.py** (4 KB)  
  The block matrices as a Word document with real, editable tables.
- **tools/build_sim_report.py** (39 KB)
- **tools/build_walk_interface.py** (63 KB)  
  A walk-through page: the view, the scores, and what the model said.
- **tools/classify_street_type.py** (6 KB)  
  Label every node vehicular or pedestrian, so the render can match the view.
- **tools/cleaned_frame_flag.py** (6 KB)  
  Mark which nodes survive the hand-checked cleaning of the frame.
- **tools/dob_sheds.py** (8 KB)  
  Sidewalk sheds and scaffolding from DOB permits, as ground truth.
- **tools/export_gis.py** (4 KB)  
  Every layer the figures draw, as one GeoPackage for QGIS or ArcGIS.
- **tools/export_svi_180.py** (15 KB)  
  Export 180-degree along-street panoramas as two walks per street.
- **tools/export_svi_90.py** (17 KB)  
  Export the forward view, split by street type: 90-degree halves or one 180.
- **tools/eyelevel.py** (12 KB)  
  Eye-level greenery against overhead canopy: the framework's first claim.
- **tools/face_samples.py** (21 KB)  
  One segmented sample per block face, with the class legend.
- **tools/factor_check.py** (7 KB)  
  Factor analysis of the directional metrics.
- **tools/flag_offstreet.py** (5 KB)  
  Flag half-views that are not street-level public space.
- **tools/fov_check.py** (10 KB)  
  Field of view as an analysis parameter: is 180 degrees the right choice?
- **tools/frame_audit.py** (3 KB)  
  Sampling-frame audit map: what is in the analytic sample and what is not.
- **tools/gwr_feasibility.py** (5 KB)  
  Can the GWR calibration in section 2.8 be estimated on this frame?
- **tools/gwr_machinery.py** (7 KB)  
  What the 26 Aug section 2.8 can be checked on, without the outcome.
- **tools/half_target.py** (6 KB)  
  Score the 90-degree half ratings against the GVI of that same 90 degrees.
- **tools/hf_bin_to_safetensors.py** (2 KB)  
  Convert a cached .bin checkpoint to safetensors, in place.
- **tools/hw_coverage_map.py** (5 KB)  
  Where the unmeasured H/W nodes actually are.
- **tools/import_frame.py** (6 KB)  
  Import an externally authored node set into the pipeline's frame schema.
- **tools/import_london_nodes.py** (5 KB)  
  Build a London frame from the colleague's node export.
- **tools/import_osm_flags.py** (6 KB)  
  Attach OSM tunnel and bridge tags to the frame, from street-view-nodes.
- **tools/import_segments.py** (6 KB)  
  Adopt the colleague's coordinate mapping as the street-segment label.
- **tools/m_maps.py** (6 KB)  
  M mapped for both study areas on ONE colour scale.
- **tools/make_model_slides.py** (17 KB)  
  Build the model-selection deck as a .pptx for import into Google Slides.
- **tools/mast_calibrate.py** (5 KB)  
  Check an imagery set's mast detection against its src/mast.py calibration.
- **tools/model_benchmark.py** (7 KB)  
  Every model on the same two targets, the same images, the same metric.
- **tools/node_map.py** (8 KB)  
  The sampling nodes on a basemap, for either study area.
- **tools/pedestrian.py** (16 KB)  
  Pedestrian-realm composition, by node and by travel direction.
- **tools/prompt_place_ab.py** (4 KB)  
  How much does the place named in the prompt move the ratings?
- **tools/rating_logits.py** (6 KB)  
  Read the model's distribution over 1-7 instead of its argmax.
- **tools/rating_logits_score.py** (4 KB)  
  Does the expected value track the street better than the argmax?
- **tools/rating_reliability.py** (5 KB)  
  Is a low-confidence field measuring the street, or rolling dice?
- **tools/renumber_svi.py** (5 KB)  
  Recompute the sequence prefix on exported half-views from the full street.
- **tools/repair_chain_pos.py** (4 KB)  
  Recompute chain_pos_m from geometry, leaving node_id untouched.
- **tools/s03_sim_profiles.py** (7 KB)  
  SIM-class azimuthal profiles: the 11 Street Interface Matrix terms.
- **tools/s03_subset.py** (10 KB)  
  Re-profile a handful of nodes from cached imagery, then diff the result.
- **tools/seg_bands.py** (6 KB)  
  Vegetation share by elevation band, so the three green fields stop sharing one twin.
- **tools/seg_combined_render.py** (8 KB)  
  One overlay per frame, each class drawn from the model the study uses for it.
- **tools/seg_gmi_band.py** (6 KB)  
  GMI's twin: greenery on the lower 3 m of the facade, per node.
- **tools/seg_two_model.py** (8 KB)  
  Per-image class shares from two segmenters: Mapillary Vistas and ADE20K.
- **tools/seg_two_model_render.py** (7 KB)  
  Render what the two segmenters actually see, side by side with the photo.
- **tools/sidewalk.py** (11 KB)  
  Sidewalk width and building setback vs directional greenness.
- **tools/sim_axonometric.py** (10 KB)  
  Exploded axonometric of the SIM layers over the built fabric.
- **tools/sim_axonometric_blender.py** (11 KB)  
  The SIM axonometric, rendered in Blender rather than drawn in matplotlib.
- **tools/sim_compute.py** (13 KB)  
  M_i for every half-view, from the VLM ratings, per manuscript section 2.7.
- **tools/sim_cube.py** (5 KB)  
  The three SIM dimensions plotted against each other, by typology.
- **tools/sim_dwell.py** (6 KB)  
  Street Interface Matrix and the demonstration dwell index.
- **tools/sim_exploded.py** (25 KB)  
  Exploded axonometric: the SIM's three dimensions as strata over the plan.
- **tools/sim_export.py** (12 KB)  
  Split the SIM run into two tables: what was observed, and what was derived.
- **tools/sim_maps.py** (7 KB)  
  I, Y, D and M mapped for both study areas, one colour scale per dimension.
- **tools/sim_readme.py** (16 KB)  
  Write the data dictionary for vlm_observations.csv and vlm_calculations.csv.
- **tools/sim_readout.py** (5 KB)  
  Turn each field's seven-rung distribution into one number, by pruning then taking the interpolated median.
- **tools/sim_samples.py** (7 KB)  
  Segmentation sample panels over the 180-degree along-street view.
- **tools/sim_section_map.py** (6 KB)  
  SIM scores aggregated to street sections, mapped and tabulated.
- **tools/sim_terms_maps.py** (7 KB)  
  The three SIM terms, one row per city.
- **tools/sim_vlm_converge.py** (9 KB)  
  Rate every half-view by elimination: prune to the rungs above chance, ask again among the survivors, repeat until one rung is left.
- **tools/sim_vlm_describe.py** (12 KB)  
  Ask the model to say what it sees, in its own words.
- **tools/sim_vlm_maps.py** (5 KB)  
  Four maps for the VLM index: M, then I_raw, Y, D_raw.
- **tools/sim_vlm_run.py** (13 KB)  
  Rate the nine SIM fields, one field per call, batched per image.
- **tools/sim_vlm_rung_ab.py** (10 KB)  
  A/B the current rungs against a rewrite, scored on the measured twin.
- **tools/sim_vlm_validate.py** (6 KB)  
  Score every VLM rating against the quantity measured over its own arc.
- **tools/sim_vlm_validate_v2.py** (6 KB)  
  Score every VLM rating against the quantity measured over its own arc.
- **tools/site_map_basemap.py** (15 KB)  
  Murray Hill on a desaturated basemap, in the reference style.
- **tools/site_map_bw.py** (7 KB)  
  Murray Hill in black and white, with the street space drawn as a figure.
- **tools/site_maps.py** (10 KB)  
  Location figures for the paper: a city locator and a Murray Hill detail.
- **tools/study_area_filter.py** (4 KB)  
  Mark which nodes are inside the defined study area.
- **tools/svi_180_comparison.py** (6 KB)  
  Photo above mask, in a tree you can arrow through like the export itself.
- **tools/svi_180_cone_sim.py** (12 KB)  
  SIM rated on three 60-degree cones, weighted as the perceptual literature.
- **tools/svi_180_qwen_morphology.py** (13 KB)  
  Qwen2-VL-7B rating micro-scale morphology and sense of place, per panorama.
- **tools/svi_180_segformer.py** (11 KB)  
  SegFormer-B5 Cityscapes over the 180-degree along-street panoramas.
- **tools/svi_180_sim_vlm.py** (14 KB)  
  The manuscript's Street Interface Matrix, asked of a VLM as judgements.
- **tools/svi_180_spotcheck.py** (7 KB)  
  Put a human in front of thirty tiles, blind, and measure the label itself.
- **tools/svi_180_vlm_scan.py** (6 KB)  
  A generative VLM's verdict on every panorama, as a score that can be ranked.
- **tools/svi_180_walk_video.py** (7 KB)  
  Each walk as a short film: the comparison frames at a fixed dwell.
- **tools/svi_90_sim.py** (12 KB)  
  SIM rated on the 90-degree halves, one row per side of each walk.
- **tools/svi_contact_sheets.py** (4 KB)  
  Contact sheets of the exported panoramas, one per street, DOB flags marked.
- **tools/svi_review_sheets.py** (3 KB)  
  Review sheets: the export in fixed-size batches, big enough to judge from.
- **tools/svi_scaffold_flag.py** (8 KB)  
  Flag which exported panoramas have a sidewalk shed in view.
- **tools/validation_figure.py** (10 KB)  
  Do the VLM's rungs correspond to anything measurable?
- **tools/vlm_benchmark.py** (11 KB)  
  Several VLMs on the same two questions, plus what other papers report.
- **tools/vmst_batch.py** (7 KB)  
  Run the segmentation-taxonomy pipeline over every svi_90 half-view.
- **tools/vmst_build.py** (4 KB)  
  Turn the Vision-Model-Segmentation-Taxonomy notebook into a batch script.
- **tools/vmst_fast.py** (5 KB)  
  Batch driver: fresh namespace per image, models loaded once.
- **tools/vmst_run.py** (168 KB)  
  GENERATED by tools/vmst_build.py -- do not edit by hand.
- **tools/walk_gif.py** (6 KB)  
  A walk down one street, as a GIF, from the rendered half-views.
- **tools/walk_seg_gif.py** (9 KB)  
  A walk down one street twice: as photographed, and as segmented.

## Drop


### figure/output, regenerates from the tools -- 81 files, 69.1 MB

- docs/index.html (2422 KB)
- results/figures/ade_signboard_low.png (713 KB)
- results/figures/bbox_extent.png (626 KB)
- results/figures/colormap_options.png (132 KB)
- results/figures/colormap_preview.png (825 KB)
- results/figures/dwell_demo.png (466 KB)
- results/figures/figure_axonometric_sim.png (901 KB)
- results/figures/figure_axonometric_sim_712.png (899 KB)
- results/figures/figure_axonometric_sim_blender.png (2913 KB)
- results/figures/figure_axonometric_sim_layers.jpg (245 KB)
- results/figures/figure_axonometric_vlm.png (857 KB)
- results/figures/figure_directional.png (845 KB)
- results/figures/figure_enclosure.png (523 KB)
- results/figures/figure_enclosure_directional.png (390 KB)
- results/figures/figure_leverage.png (264 KB)
- results/figures/figure_maps.png (853 KB)
- results/figures/figure_rose.png (898 KB)
- results/figures/figure_scatter.png (743 KB)
- results/figures/figure_site_basemap_context.pdf (472 KB)
- results/figures/figure_site_basemap_context.png (6424 KB)
- results/figures/figure_site_basemap_context_bw.pdf (320 KB)
- results/figures/figure_site_basemap_context_bw.png (5675 KB)
- results/figures/figure_site_basemap_detail.pdf (389 KB)
- results/figures/figure_site_basemap_detail.png (4451 KB)
- results/figures/figure_site_basemap_detail_bw.pdf (265 KB)
- results/figures/figure_site_basemap_detail_bw.png (3708 KB)
- results/figures/figure_site_locator.pdf (816 KB)
- results/figures/figure_site_locator.png (407 KB)
- results/figures/figure_site_murrayhill.pdf (178 KB)
- results/figures/figure_site_murrayhill.png (2014 KB)
- results/figures/figure_site_murrayhill_bw.pdf (201 KB)
- results/figures/figure_site_murrayhill_bw.png (1267 KB)
- results/figures/figure_site_murrayhill_bw_outline.pdf (207 KB)
- results/figures/figure_site_murrayhill_bw_outline.png (1590 KB)
- results/figures/figure_site_murrayhill_figureground.pdf (178 KB)
- results/figures/figure_site_murrayhill_figureground.png (1460 KB)
- results/figures/figure_site_murrayhill_grey.pdf (178 KB)
- results/figures/figure_site_murrayhill_grey.png (1620 KB)
- results/figures/frame_audit.png (186 KB)
- results/figures/frame_compare.png (881 KB)
- results/figures/frame_nodes_compare.png (816 KB)
- results/figures/hw_coverage_map.png (657 KB)
- results/figures/mast_detect.png (609 KB)
- results/figures/mast_detect2.png (516 KB)
- results/figures/mast_fixed.png (920 KB)
- results/figures/mast_fixed2.png (422 KB)
- results/figures/mast_overmask.png (1045 KB)
- results/figures/method_band_mechanics.png (215 KB)
- results/figures/method_band_min_failure.png (262 KB)
- results/figures/method_cone_vs_band.png (344 KB)
- results/figures/method_coverage_now.png (981 KB)
- results/figures/method_crossing_guard.png (224 KB)
- results/figures/method_donor_radius.png (96 KB)
- results/figures/method_e40_break.png (458 KB)
- results/figures/method_none_nodes.png (273 KB)
- results/figures/method_open_audit.png (557 KB)
- results/figures/method_probe_cases.png (305 KB)
- results/figures/method_probe_reach.png (171 KB)
- results/figures/method_segment_median.png (237 KB)
- results/figures/model_selection_slides.pptx (43 KB)
- ... and 21 more

### experiment or duplicate of a final table -- 58 files, 10.8 MB

- cubemap_check.py (12 KB)
- migrate_gridaxis.py (5 KB)
- results/london/tables/sim_vlm_london_converged.csv (1549 KB)
- results/tables/anchor_probe.csv (3 KB)
- results/tables/describe_prompt_ab.csv (8 KB)
- results/tables/describe_vs_score.csv (0 KB)
- results/tables/openvocab_eval.csv (5 KB)
- results/tables/probe_band_median_vs_cone.csv (76 KB)
- results/tables/probe_band_vs_cone.csv (122 KB)
- results/tables/probe_variants.pkl (192 KB)
- results/tables/probe_vei_180.csv (6 KB)
- results/tables/probe_vei_180_summary.csv (0 KB)
- results/tables/probe_vei_90.csv (6 KB)
- results/tables/probe_vei_90_summary.csv (0 KB)
- results/tables/prompt_probe.csv (19 KB)
- results/tables/prompt_probe_180.csv (12 KB)
- results/tables/prompt_probe_90_noside.csv (5 KB)
- results/tables/prompt_probe_90_noside_summary.csv (0 KB)
- results/tables/prompt_probe_summary.csv (2 KB)
- results/tables/scaffold_eval.csv (4 KB)
- results/tables/scale_probe.csv (30 KB)
- results/tables/sim_vlm_180_holdout.csv (230 KB)
- results/tables/sim_vlm_converged.csv (2471 KB)
- results/tables/sim_vlm_v3_readout.csv (826 KB)
- results/tables/svi_180_cone_eval.csv (3 KB)
- results/tables/svi_180_cone_test.csv (28 KB)
- results/tables/svi_180_projection_test.csv (11 KB)
- results/tables/vlm_calculations_murrayhill.csv (1654 KB)
- results/tables/vlm_describe_probe.csv (120 KB)
- results/tables/vlm_mast_probe.csv (7 KB)
- results/tables/vlm_mast_prompt_probe.csv (10 KB)
- results/tables/vlm_observations_murrayhill.csv (3080 KB)
- results/tables/vlm_reask_probe.csv (178 KB)
- results/tables/vlm_reask_probe_oldrungs.csv (233 KB)
- tools/anchor_probe.py (7 KB)
- tools/batch_probe.py (7 KB)
- tools/batch_size_probe.py (4 KB)
- tools/cubemap_check.py (12 KB)
- tools/describe_prompt_ab.py (7 KB)
- tools/describe_vs_score.py (10 KB)
- tools/facade_axis_test.py (5 KB)
- tools/openvocab_eval.py (10 KB)
- tools/probe_provenance.py (2 KB)
- tools/prompt_probe.py (15 KB)
- tools/scaffold_eval.py (7 KB)
- tools/scale_probe.py (5 KB)
- tools/scale_probe_score.py (3 KB)
- tools/sim_vlm_describe_probe.py (13 KB)
- tools/sim_vlm_reask_probe.py (11 KB)
- tools/survey_probe.py (5 KB)
- tools/svi_180_cone_eval.py (9 KB)
- tools/svi_180_cone_test.py (9 KB)
- tools/svi_180_probe_eval.py (8 KB)
- tools/svi_180_probe_features.py (7 KB)
- tools/svi_180_projection_test.py (8 KB)
- tools/vlm_mast_probe.py (6 KB)
- tools/vlm_mast_prompt_probe.py (6 KB)
- tools/width_probe_diagram.py (12 KB)

### this machine's overnight scheduling, absolute paths -- 5 files, 0.0 MB

- handover_0200.sh (5 KB)
- run_london_queue.sh (3 KB)
- run_london_queue2.sh (2 KB)
- run_vlm_london.sh (1 KB)
- supervise.sh (2 KB)
