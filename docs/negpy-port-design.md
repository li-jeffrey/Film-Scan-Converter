# Design Plan: Port NegPy's negative conversion engine into Film Scan Converter

## Context

Film Scan Converter (FSC, `/Users/jeffrey.li/personal/film-scan-converter`) converts negatives with crude math. `source/RawProcessing.py` decodes the RAW with a BT.709 gamma and +3 EV into P3. It then inverts with a linear `65535 - x`, applies a per-channel percentile levels stretch (`hist_EQ`, :477), applies multiplicative temp/tint (`wb_adjust_coeff`, :529), then gamma, shadows and highlights (`exposure`, :578), then HSV saturation through matplotlib (:598). It has no density-domain processing, no film/paper model, and no color management on export.

NegPy (`/Users/jeffrey.li/personal/negpy`) has a physically motivated engine:
- scene-linear decode, log10 density, dye-crosstalk unmix;
- robust luma/color-split bounds metering on a block-median grid;
- an H&D print curve with softplus toe and shoulder, ISO-R grade, auto density and grade, Cast Removal from a neutral-axis meter, CMY filtration;
- paper profiles and a slide transfer path.

Both projects are GPL-3, so copying code is allowed. Add a NegPy credit to the README/NOTICE.

**Decided scope:**
- Port the core conversion, crosstalk and paper profiles.
- Replace the GUI's Levels/Exposure/Colour controls with NegPy-style controls.
- Port the kernels as vectorized NumPy, with no numba.
- Out of scope: roll-level pooled bounds, the Lab stage (CLAHE, sharpening, gamut saturation), GPU, IR dust, and NegPy's autocrop and geometry. FSC keeps its own crop, dust, rotate, frame and aspect code.

---

## 1. New package: `source/conversion/`

Pure functions on RGB float32 arrays. No Tk, no `self`, picklable, and laid out file-for-file like NegPy so future upstream changes can be diffed.

| File | Contents |
|---|---|
| `constants.py` | `EXPOSURE_CONSTANTS` (negpy `features/exposure/models.py:120-281`), transfer constants (`transparency/logic.py:55-92`), Rec.709 luma, Adobe RGB and sRGB matrices |
| `params.py` | Frozen `ConversionParams`, `from_photo(photo)`, slider→param mappings, a `SPECS` table (key, label, range, default, widget), `migrate_v1()` |
| `decode.py` | Linear rawpy decode, linearization of non-RAW/Lightroom input, display LUT |
| `icc.py` | Minimal matrix/TRC ICC parser (`r/g/bTRC`, `r/g/bXYZ`), bundled ICC bytes, PNG `iCCP` injector |
| `density.py` | log10, normalize, `_block_median_grid`, bounds, all meters |
| `crosstalk.py` | TOML loader (`tomllib`), resolve, process gate, unmix |
| `papers.py` | Paper profiles |
| `print_curve.py` | Grade/pivot/Cast Removal solve, auto helpers, vectorized print curve, WB/Kelvin helpers |
| `transfer.py` | Slide path: camera matrix and transfer curve with the ACES ODT |
| `output.py` | Working OETF (x^(256/563)), Adobe→sRGB, encode to uint16/uint8 |
| `pipeline.py` | `build_grid()`, `analyze(grid, params) → Analysis`, `normalize()`, `render(norm, params, analysis, out_space)` |

Data goes in `source/assets/crosstalk/*.toml` (16 files from `negpy/crosstalk/`) and `source/assets/icc/` (sRGB-v4, AdobeCompat-v4, GrayGamma2.2 from `negpy/icc/`). The existing `--add-data assets` in `scripts/build.ps1` and `ResourcePaths.resource_path` bundle them with no build change.

## 2. Porting map (NegPy → new file)

Paths are relative to `negpy/negpy/`.

| NegPy source | Target | Action |
|---|---|---|
| `features/exposure/normalization.py:27-53,720` `_normalize_log_image_jit` | density | Vectorize `(x-f)/(c-f)` |
| `:66-107` analysis region / buffer | — | Drop. FSC `crop(..., include_EQ_ignore=True)` defines the region |
| `:110-148` `_block_median_grid`; `:151-193` prefilter, clip fractions | density | Port without the thread pool. Clip fraction becomes a logged warning |
| `:196-237` crosstalk resolve/gate/unmix | crosstalk | Port, using chunked `reshape(-1,3) @ M.T` |
| `:240-626` sorted grid, percentiles, `_rms_chroma`, shadow refs, neutral axis, `_textured_cells`, anchor, textural range, shadow/highlight points | density | Port as-is |
| `:730-922` `_sample_log_bounds`, `_same_pixel_color_floor_refs`, `analyze_log_exposure_bounds_from_log` | density | Port as-is, plus an optional `thin_end_override` for the manual base (§5) |
| `:653-717` contrast mask; `:925-1083` pool/mix/resolve bounds | — | Drop (roll pooling is out of scope) |
| `features/exposure/processor.py:52-189,225-391` | pipeline | Port the stage order. Drop locks, local maps, contrast mask, ev_map, histogram metrics |
| `features/exposure/logic.py:19-53` expit/softplus/inverse softplus; `:101` `quadratic_core` | print_curve | `np.logaddexp(0,x)` for softplus; vertex clamp with `np.where` |
| `:118-352` `_apply_print_curve_kernel` | print_curve `apply_print_curve` | Rebuild from the already-NumPy `CharacteristicCurve.__call__` (`:424-454`), then the dye-mix matmul, `10**-D`, BPC and clip. Row-chunked float32. Drop the local ev/grade maps, split grade, regional CMY, preflash and separation damping |
| `:558-613,616-761` per-channel toe/shoulder, `apply_characteristic_curve` | print_curve | Keep without trims or local inputs |
| `:793-853,935-1288` `grade_to_slope`, `grade_coupled_shape`, `effective_grade_range`, `_reference_linear_value`, shadow reach, highlight hold, `compute_pivot`, `cast_solve_inputs`, `per_channel_curve_params` | print_curve | Port as-is; `grade_trims` = 0. Add `slope_to_grade` for the read-out |
| `:1338-1354,1433-1516` `filtration_offsets`, CMY↔density, `wb_to_kelvin`/`kelvin_to_wb`, `calculate_wb_shifts_from_log` | print_curve | Port as-is |
| `:764-790,888-932,1002-1018,1357-1430` flat curve, local/split grade, preflash, local EV | — | Drop |
| `features/exposure/models.py:21-117` `ExposureConfig` | params | Subset only. Drop `TUNABLE_TARGETS` and `RenderIntent` |
| `features/process/models.py` `ProcessMode`, `cast_removal_for_mode`, `auto_meter_for_mode`, `DEFAULT_CROSSTALK_MATRIX`, `per_channel_point_offsets` | params / crosstalk | Port (global offsets only) |
| `features/exposure/papers.py:22-274` | papers | Port as-is; drop `lith_path` |
| `features/transparency/logic.py:94-473`, `processor.py:41-201`; `features/process/capture_color.py:27-137` | transfer | Port (already NumPy): `camera_to_working_matrix`, `apply_camera_matrix`, `apply_transfer_curve` |
| `services/rendering/image_processor.py:880-956`, `infrastructure/loaders/helpers.py:616-650`, `raw_demosaic.py:22-55` | decode | Decode kwargs, `_user_sat`/linearity limit, camera xyz matrix, WB multipliers |
| `kernel/image/logic.py:135-183,630` | output | `srgb_to_linear`, working OETF, luminance |
| `services/assets/crosstalk.py:46-165` | crosstalk | TOML scan/parse. A missing `process` key means C-41 |
| `desktop/view/sidebar/process.py:24-72` | params | Luma/color clip slider maps |

## 3. New `RawProcessing.process` flow and caching

**Buffers on `RawProcessing`:**
- `RAW_LIN`: linear uint16 RGB from the decoder.
- `RAW_DISP`: uint8 BGR display copy made with a 65536-entry LUT (camera-WB normalize → 0.95/P99.5 gain → BT.709 OETF). It feeds `find_optimal_crop`, the threshold/contour views, `find_dust`, the RAW view and the base picker, so those functions stay unchanged.
- `ANALYSIS_SRC`: `RAW_LIN` resized to a 2048 px long edge with INTER_AREA, built once per load. Preview and export therefore meter identically.
- `IMG`: uint16 BGR, display/output-encoded. `fill_dust`, `add_frame`, `rotate`, `draw_histogram` and the cv2 paths keep working.

**`process(full_res, recent_only, skip_crop, for_export=False)`:**
1. If needed, run crop detection on `RAW_DISP` and build the proxies (`proxy_LIN`, `proxy_DISP`), as today (:287-303).
2. `lin = crop(src_lin, rect)`. **Crop before converting**: every per-pixel step is position-independent, so the result is the same and the work is smaller.
3. `dust_mask = find_dust(crop(src_disp, rect))`.
4. Grid (cached): `crop(ANALYSIS_SRC, rect, include_EQ_ignore=True)` → camera matrix (slides only) → log10 → unmix → block median → sort.
5. Analysis (cached): bounds (with optional base override) → point offsets → shadow refs, neutral axis, anchor, textural range, shadow/highlight points. Wrap in a `threading.Lock`, because the GUI thread and the full-preview thread both call `process`.
6. `norm = normalize(unmix(log10(lin)), bounds)`, cached for the proxy only.
7. `render` in row chunks of about 1M px, writing straight into a preallocated uint16 output (sRGB for preview; the Advanced output space when `for_export`). Swap RGB→BGR only at this boundary.

**What each change invalidates** (`RawProcessing` compares cache keys itself, so the GUI's widget modes barely change):

| Change | Crop | Grid | Analysis | Norm | Curve |
|---|---|---|---|---|---|
| thresholds, `border_crop` | ✓ | ✓ | ✓ | ✓ | ✓ |
| analysis border, `film_type`, crosstalk stock/strength | | ✓ | ✓ | ✓ | ✓ |
| luma/color clip, film base | | | ✓ | ✓ | ✓ |
| white/black point offsets | | | bounds shift only | ✓ | ✓ |
| density, grade, autos, toe/shoulder, zones, paper, Cast Removal, CMY/temp, saturation | | | | | ✓ |
| rotation, flip, frame, dust toggle | render only (existing `'update'` mode) | | | | |

**Pickling for batch export** (`GUI.py:231-350`): `__getstate__` drops the ndarrays, grid, norm cache and lock but keeps the small `Analysis` tuple, so exports use the bounds the preview showed. `__setstate__` recreates the lock. Recompute `memory_alloc` (:124) as about `RAW_LIN.nbytes * 5`.

## 4. Decode and input changes (`decode.py`, `RawProcessing.load`, `RawProcessing.py:81-154`)

**RAW decode:**
- rawpy settings: `gamma=(1,1)`, `no_auto_bright=True`, `adjust_maximum_thr=0`, `user_sat=_user_sat(raw)`, `output_color=raw`, `output_bps=16`, `user_wb=[1,1,1,1]`.
- Keep `dm_alg`, `half_size`, `fbdd_nr`, `median_filter_passes`, `noise_thr`. Keep libraw's default flip, because saved `rotation` values depend on it.
- Store `camera_wb` and `rgb_xyz_matrix` on the instance.
- **WB handling:** decode once with unity WB.
  - Negatives / B&W: multiply by the as-shot camera WB in linear after decode. This matches NegPy's default `linear_raw=False` and keeps the chroma gating in the meters calibrated.
  - Slides: use unity WB with the camera matrix folding WB back in (NegPy `effective_linear_raw`).
  - A diagonal gain commutes with decode, so changing film type never re-decodes.
- The black-border trim (:115-121) runs on the display copy. Keep the data RGB (drop the `::-1`).

**Non-RAW files and the Lightroom TIFF:**
- Read with tifffile/PIL and get the ICC (tag 34675).
- `icc.py` extracts the TRC (`curv` gamma/table, `para` 0-4) to linearize. Slides also get the primaries→Adobe RGB matrix (Bradford D50→D65).
- If there is no ICC: 8-bit is treated as sRGB, 16-bit follows the Advanced "Input encoding" (Auto / sRGB / Linear).
- Lightroom output is always Adobe RGB 16-bit with AdobeCompat embedded. This replaces re-embedding the source ICC (:252-266).

## 5. Film-type handling

| Type | Path |
|---|---|
| Colour neg (1) | Print path, C-41: crosstalk, Cast Removal 1.0, auto density and grade on, Neutral/RA4 papers |
| B&W (0) | Luma of `norm` → single-channel curve (3× cheaper) → 2-D gray out. No crosstalk or Cast Removal, color frame hidden, B&W papers (Ilford MG RC/FB, Foma) |
| Slide (2) | `camera_to_working_matrix` → log → fixed window (0, −3 D) → transfer curve + ACES ODT. Cast Removal 0, autos off, no crosstalk, clip sliders hidden. Non-RAW input skips the matrix |
| Crop only (3) | Cropped `RAW_LIN` through the display transform at 16 bits, no ICC (camera space) |

**Film base / Import Blank:**
- Kept for colour negatives only, as a **thin-end bounds override**: `ceils = unmix(log10(base_lin))`, and the dense-end `_same_pixel_color_floor_refs` is anchored to it.
- `base_rgb` becomes a linear float triple. `get_base_colour` (:722) samples `RAW_LIN`, and Import Blank (`BaseGUI.py:884-900`) uses the linear P90.
- The swatch shows `LUT(base)`. Legacy 8-bit values reset to Auto.

## 6. GUI (`BaseGUI.py`: default_settings :47-64, widgets :121-205, update_UI :714-752, Advanced dialog :412-446)

The new controls replace the White/Black Point, Gamma, Shadows/Highlights, Temp/Tint and Saturation widgets.

| Label | Key | Range / step | Default | NegPy param |
|---|---|---|---|---|
| Print Density | `density` | 0–2 / 0.05 | 1.0 | density |
| Grade (ISO R) | `grade` | 50–180 / 1 | 115 | grade |
| Auto Density / Auto Grade | `auto_density`, `auto_grade` | check | on (neg), off (slide) | auto_exposure / auto_normalize_contrast |
| Toe, Shoulder | `toe`, `shoulder` | −1–1 / 0.05 | 0 | same |
| Toe/Shoulder Width | `toe_width`, `shoulder_width` | 0.1–5 / 0.1 | 2.5 | same |
| Shadows / Highlights | `shadow_density`, `highlight_density` | ±0.9 / ±0.5 | 0 | zone density |
| Paper | `paper_profile` | combo, filtered by mode | neutral | paper profile |
| Paper White / Paper Black | `paper_dmin`, `paper_black` | check | off | d_min / BPC off |
| Temperature (K) | derived, not stored | 3000–12000 | 5500 | `kelvin_to_wb` → M/Y |
| Cyan / Magenta / Yellow | `wb_cyan`, `wb_magenta`, `wb_yellow` | −1–1 / 0.01 | 0 | `filtration_offsets` |
| WB picker | writes M/Y | button | — | `calculate_wb_shifts_from_log` on the cached norm, same circular mask as :533-544 |
| Saturation | `dye_separation` | 0.5–1.5 / 0.05 | 1.0 | dye separation |
| Cast Removal | `cast_removal` | 0–1 / 0.05 | 1.0 (neg), 0 (slide) | strength × confidence |
| Film Stock | `crosstalk_profile` | Generic C41 + 16 TOMLs | Generic C41 | matrix |
| Crosstalk Strength | `crosstalk_strength` | 0–1 / 0.05 | 0.0 | blend with identity |
| White / Black Point | `white_point_offset`, `black_point_offset` | ±0.25 / 0.01 | 0 | point offsets |
| Luma Clip / Colour Clip | `luma_clip_pos`, `color_clip_pos` | −100–100 | 0 | NegPy slider maps |
| Film Base | `base_detect`, `base_rgb` | existing combo + picker | Auto | thin-end override |

Other changes:
- **Effective grade:** a read-only "Effective grade R…" label (from `slope_to_grade`).
- **Film type switch:** changing `film_type` applies the mode defaults (`cast_removal_for_mode`, `auto_meter_for_mode`) and refreshes the paper list.
- **Advanced dialog:**
  - Remove `colour_space`, `raw_gamma`, `exp_shift`, `use_camera_wb`, `wb_mult`, and the black/white point percentiles.
  - Relabel `ignore_border` to "Analysis Border %" with default (5,5), matching NegPy's 0.05 buffer.
  - Add `output_colour_space` (sRGB / Adobe RGB) and `input_encoding`.
- **Migration:**
  - The per-photo `.npy` gets `settings_version: 2`. For v1 files, keep the geometry, `film_type`, `reject` and `remove_dust` keys, set `dye_separation = clip(sat/100, 0.5, 1.5)`, and take everything else from defaults.
  - `config.npy` gets `config_version`. If it is absent, drop the stale `ignore_border`.
  - Also fix the load/save filename mismatch (`split('.')[0]` vs `splitext`, `RawProcessing.py:59-79,268-274`).

## 7. Output and dependencies

- `render` returns linear Adobe-RGB-primaries data.
- **Export spaces:**
  - sRGB (default): matrix, sRGB OETF, embed sRGB-v4.icc.
  - Adobe RGB: x^(256/563), embed AdobeCompat-v4.
  - B&W: gray γ2.2 with GrayGamma2.2.icc.
- **Writers:**
  - JPG: PIL with `icc_profile`.
  - TIFF: tifffile with `iccprofile`.
  - PNG: cv2 16-bit, then inject `iCCP`.
- **Preview:** always sRGB.
- **requirements.txt:** remove matplotlib (only used at :580-607). Collapse the three opencv wheels to `opencv-python-headless`. Verify rawpy 0.24 exposes `rgb_xyz_matrix`, `camera_whitebalance` and `camera_white_level_per_channel`, and bump it if not.

## 8. Testing (new; FSC has none)

- **Setup:** `pyproject.toml` gets `[tool.pytest.ini_options] pythonpath=["source"]`, with `tests/` and `tests/parity/`.
- **Ported NegPy unit tests** (from `negpy/tests/`, pure-logic parts only): characteristic curve, exposure logic/pivot, grade calibration, robust/same-pixel bounds, neutral axis, luma range margin, color/luma split, auto exposure/contrast, paper profiles/black/dmin/S-curve, toe-shoulder, working OETF, E-6 mode, crosstalk profiles, transparency transfer, slide cast removal, dye separation. Skip `test_scene_linear_relocation`, whose goldens include the Lab stage.
- **Parity:**
  - `scripts/make_negpy_fixtures.py` is run with `uv run --project ../negpy`. It writes `tests/fixtures/negpy_parity.npz` (bounds, meters, curve params, curve and transfer outputs on synthetic inputs).
  - Optional live tests use `NEGPY_PATH` with `pytest.importorskip`.
  - Tolerances: 1e-6 for meters, 1e-4 for the curve (float32 vs NegPy float64).
- **Smoke test:** a synthetic orange-masked negative (gray ramp + patches, D = γ·(−log s) + base ≈ (0.25, 0.6, 0.9)) run through `pipeline` for C-41 and B&W. Assert no NaNs, gray-ramp chroma < 0.02, monotonic, and hue order preserved.
- **Integration:** a synthetic 16-bit TIFF → `RawProcessing.load/process/export` for each format. Check the ICC is present and that the pickled size is under 1 MB.
- **Benchmark:** a slow-marked 24 MP render benchmark.
- **Manual:** run `python source/App.pyw <scans>` on real C-41, B&W and slide RAWs. Check crop detection against v0.2.5, preview latency under ~150 ms per slider move, the WB picker, Import Blank, and batch export.

## 9. Phases

| Phase | Work | Checkpoint |
|---|---|---|
| P0 | pytest setup, package skeleton, `constants.py`, fixture script | — |
| P1 | `density.py` + `crosstalk.py` | Bounds and meter parity green |
| P2 | `papers.py` + `print_curve.py` | Curve parity green |
| P3 | `pipeline.py` + `output.py` | Synthetic smoke test green |
| P4 | `decode.py`, display LUT, `RawProcessing` integration (process order, caches, views, dust, Lightroom) | Real scans render; crop thresholds re-tuned if needed |
| P5 | `transfer.py` (slides), B&W, crop-only | Slide parity green |
| P6 | GUI controls, mode defaults, WB picker, base override, Advanced dialog, migration | Manual GUI pass |
| P7 | ICC export, batch pickling, requirements cleanup, docs/CHANGELOG, NegPy credit | Integration tests green |

## Risks

- **Full-resolution NumPy speed:** about 30 ops per element over 72M elements takes seconds. Mitigate with ~1M-px chunks, skipping zero-valued terms, and threads in the GUI process only. Numba remains a later option.
- **Memory:** never hold a full-res float32 output; render chunks straight into the uint16 output.
- **Crop thresholds:** the defaults (25/100) were tuned on the old +3 EV gamma decode, so the new display LUT may need retuning.
- **Pickling:** the lock and caches must be excluded from pickling, and `class_parameters` still has to be re-injected in the workers.
