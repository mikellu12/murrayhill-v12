
===== pipeline stage  (13) =====
  export_gis.py                     Every layer the figures draw, as one GeoPackage for QGIS or ArcGIS.
  export_svi_180.py                 Export 180-degree along-street panoramas as two walks per street.
  export_svi_90.py                  Export the forward view, split by street type: 90-degree halves or one 180.
  s03_sim_profiles.py               SIM-class azimuthal profiles: the 11 Street Interface Matrix terms.
  s03_subset.py                     Re-profile a handful of nodes from cached imagery, then diff the result.
  seg_bands.py                      Vegetation share by elevation band, so the three green fields stop sharing one twin.
  seg_combined_render.py            One overlay per frame, each class drawn from the model the study uses for it.
  seg_gmi_band.py                   GMI's twin: greenery on the lower 3 m of the facade, per node.
  seg_two_model.py                  Per-image class shares from two segmenters: Mapillary Vistas and ADE20K.
  seg_two_model_render.py           Render what the two segmenters actually see, side by side with the photo.
  sim_vlm_converge.py               Rate every half-view by elimination: prune to the rungs above chance, ask again among th
  sim_vlm_describe.py               Ask the model to say what it sees, in its own words.
  sim_vlm_run.py                    Rate the nine SIM fields, one field per call, batched per image.

===== analysis / output  (24) =====
  build_sim_report.py               
  build_walk_interface.py           A walk-through page: the view, the scores, and what the model said.
  flag_offstreet.py                 Flag half-views that are not street-level public space.
  node_map.py                       The sampling nodes on a basemap, for either study area.
  node_usability.py                 usable: True/False on nodes.csv, instead of a node quietly not being there.
  sim_axonometric.py                Exploded axonometric of the SIM layers over the built fabric.
  sim_axonometric_blender.py        The SIM axonometric, rendered in Blender rather than drawn in matplotlib.
  sim_compute.py                    M_i for every half-view, from the VLM ratings, per manuscript section 2.7.
  sim_cube.py                       The three SIM dimensions plotted against each other, by typology.
  sim_dwell.py                      Street Interface Matrix and the demonstration dwell index.
  sim_exploded.py                   Exploded axonometric: the SIM's three dimensions as strata over the plan.
  sim_export.py                     Split the SIM run into two tables: what was observed, and what was derived.
  sim_maps.py                       I, Y, D and M mapped for both study areas, one colour scale per dimension.
  sim_readme.py                     Write the data dictionary for vlm_observations.csv and vlm_calculations.csv.
  sim_readout.py                    Turn each field's seven-rung distribution into one number, by pruning then taking the in
  sim_samples.py                    Segmentation sample panels over the 180-degree along-street view.
  sim_section_map.py                SIM scores aggregated to street sections, mapped and tabulated.
  sim_terms_maps.py                 The three SIM terms, one row per city.
  sim_vlm_maps.py                   Four maps for the VLM index: M, then I_raw, Y, D_raw.
  sim_vlm_validate.py               Score every VLM rating against the quantity measured over its own arc.
  sim_vlm_validate_v2.py            Score every VLM rating against the quantity measured over its own arc.
  validation_figure.py              Do the VLM's rungs correspond to anything measurable?
  walk_gif.py                       A walk down one street, as a GIF, from the rendered half-views.
  walk_seg_gif.py                   A walk down one street twice: as photographed, and as segmented.

===== probe / one-off  (28) =====
  anchor_probe.py                   Two phrasings each for the three fields that stayed flat. Two minutes.
  batch_probe.py                    Does asking the nine fields as a batch cost less than nine calls?
  batch_size_probe.py               How many images can share one generate() call before the card runs out?
  cubemap_check.py                  Cubemap vs 4-image sampling: how much does the missing sky bias GVI and VEI?
  describe_prompt_ab.py             Does enumerating the options make the model invent them?
  facade_axis_test.py               Does the cleaned-frame street axis recover the nodes s05 loses?
  factor_check.py                   Factor analysis of the directional metrics.
  fov_check.py                      Field of view as an analysis parameter: is 180 degrees the right choice?
  openvocab_eval.py                 Per-class validation: does open-vocabulary grounding actually see it?
  probe_provenance.py               Who took each panorama: Google, or a user.
  prompt_place_ab.py                How much does the place named in the prompt move the ratings?
  prompt_probe.py                   Is the twelve-field schema what is flattening the SIM ratings?
  renumber_svi.py                   Recompute the sequence prefix on exported half-views from the full street.
  scaffold_eval.py                  Does the scaffolding detector agree with the permits? A balanced test.
  scale_probe.py                    Two anchors against seven, read as argmax and as expected value.
  scale_probe_score.py              Four cells -- two anchors or seven, argmax or expected value -- scored.
  sim_vlm_describe_probe.py         Does giving the model room to think before it answers change the answer?
  sim_vlm_reask_probe.py            Prune the scale to the bins the model favours, then ask it again.
  sim_vlm_rung_ab.py                A/B the current rungs against a rewrite, scored on the measured twin.
  survey_probe.py                   Ask the model like a survey: many respondents, one number each.
  svi_180_cone_eval.py              Which weighting of the three cones best reproduces the VLM's own whole-view judgement?
  svi_180_cone_test.py              Does the VLM already weight the centre of a view, or does it read it flat?
  svi_180_probe_eval.py             Score candidate scaffolding signals against the labels, not the permits.
  svi_180_probe_features.py         Two measurements over the 180-degree panoramas, one pass, no wiring.
  svi_180_projection_test.py        Does the cylindrical reprojection cost the VLM anything?
  vlm_mast_probe.py                 Does the VLM see Google's camera mast as part of the street?
  vlm_mast_prompt_probe.py          Can one sentence do what erasing pixels does?
  width_probe_diagram.py            How the street width is measured, and the two probes that came before.

===== other  (51) =====
  anchor_score.py                   Score anchor_probe.csv against the measured share over each 90-degree arc.
  block_counts.py                   Node counts per street and per street block.
  block_matrix.py                   The cross-streets as a matrix: 9 streets by 5 blocks, laid out as the map.
  block_matrix_docx.py              The block matrices as a Word document with real, editable tables.
  classify_street_type.py           Label every node vehicular or pedestrian, so the render can match the view.
  cleaned_frame_flag.py             Mark which nodes survive the hand-checked cleaning of the frame.
  describe_vs_score.py              Do the model's words agree with the model's numbers?
  dob_sheds.py                      Sidewalk sheds and scaffolding from DOB permits, as ground truth.
  eyelevel.py                       Eye-level greenery against overhead canopy: the framework's first claim.
  face_samples.py                   One segmented sample per block face, with the class legend.
  frame_audit.py                    Sampling-frame audit map: what is in the analytic sample and what is not.
  gwr_feasibility.py                Can the GWR calibration in section 2.8 be estimated on this frame?
  gwr_machinery.py                  What the 26 Aug section 2.8 can be checked on, without the outcome.
  half_target.py                    Score the 90-degree half ratings against the GVI of that same 90 degrees.
  hf_bin_to_safetensors.py          Convert a cached .bin checkpoint to safetensors, in place.
  hw_coverage_map.py                Where the unmeasured H/W nodes actually are.
  import_frame.py                   Import an externally authored node set into the pipeline's frame schema.
  import_london_nodes.py            Build a London frame from the colleague's node export.
  import_osm_flags.py               Attach OSM tunnel and bridge tags to the frame, from street-view-nodes.
  import_segments.py                Adopt the colleague's coordinate mapping as the street-segment label.
  m_maps.py                         M mapped for both study areas on ONE colour scale.
  make_model_slides.py              Build the model-selection deck as a .pptx for import into Google Slides.
  mast_calibrate.py                 Check an imagery set's mast detection against its src/mast.py calibration.
  model_benchmark.py                Every model on the same two targets, the same images, the same metric.
  pedestrian.py                     Pedestrian-realm composition, by node and by travel direction.
  rating_logits.py                  Read the model's distribution over 1-7 instead of its argmax.
  rating_logits_score.py            Does the expected value track the street better than the argmax?
  rating_reliability.py             Is a low-confidence field measuring the street, or rolling dice?
  repair_chain_pos.py               Recompute chain_pos_m from geometry, leaving node_id untouched.
  sidewalk.py                       Sidewalk width and building setback vs directional greenness.
  site_map_basemap.py               Murray Hill on a desaturated basemap, in the reference style.
  site_map_bw.py                    Murray Hill in black and white, with the street space drawn as a figure.
  site_maps.py                      Location figures for the paper: a city locator and a Murray Hill detail.
  study_area_filter.py              Mark which nodes are inside the defined study area.
  svi_180_comparison.py             Photo above mask, in a tree you can arrow through like the export itself.
  svi_180_cone_sim.py               SIM rated on three 60-degree cones, weighted as the perceptual literature.
  svi_180_qwen_morphology.py        Qwen2-VL-7B rating micro-scale morphology and sense of place, per panorama.
  svi_180_segformer.py              SegFormer-B5 Cityscapes over the 180-degree along-street panoramas.
  svi_180_sim_vlm.py                The manuscript's Street Interface Matrix, asked of a VLM as judgements.
  svi_180_spotcheck.py              Put a human in front of thirty tiles, blind, and measure the label itself.
  svi_180_vlm_scan.py               A generative VLM's verdict on every panorama, as a score that can be ranked.
  svi_180_walk_video.py             Each walk as a short film: the comparison frames at a fixed dwell.
  svi_90_sim.py                     SIM rated on the 90-degree halves, one row per side of each walk.
  svi_contact_sheets.py             Contact sheets of the exported panoramas, one per street, DOB flags marked.
  svi_review_sheets.py              Review sheets: the export in fixed-size batches, big enough to judge from.
  svi_scaffold_flag.py              Flag which exported panoramas have a sidewalk shed in view.
  vlm_benchmark.py                  Several VLMs on the same two questions, plus what other papers report.
  vmst_batch.py                     Run the segmentation-taxonomy pipeline over every svi_90 half-view.
  vmst_build.py                     Turn the Vision-Model-Segmentation-Taxonomy notebook into a batch script.
  vmst_fast.py                      Batch driver: fresh namespace per image, models loaded once.
  vmst_run.py                       GENERATED by tools/vmst_build.py -- do not edit by hand.
