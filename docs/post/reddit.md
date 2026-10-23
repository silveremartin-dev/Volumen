# Reddit launch post — Volumen

## Suggested subreddits

- **r/VesuviusChallenge** (and the Scroll Prize Discord, which is where most of the community actually is): the main audience. They know `vc_render`, VC3D and tifxyz far better than anyone, so lead with questions.
- **r/rust**: crate layout, out-of-core Zarr with a bounded LRU, Rayon sampling, PyO3 bindings. Expect (and welcome) hard questions on performance claims.
- **r/DigitalHumanities** / **r/AncientRome** (optional): a non-technical version with the Scroll 1 comparison image.
- **r/computervision** (optional): surface-volume sampling and normal estimation on noisy meshes.

---

**Title:** Volumen: an early-stage Rust engine for rendering Herculaneum scroll surface volumes (OME-Zarr out-of-core, .obj + tifxyz, PyO3). Looking for feedback from the Vesuvius community

---

Hi all,

I've open-sourced **Volumen**, an early-stage Rust workspace aimed at one narrow part of the Vesuvius Challenge pipeline: **turning a segmented sheet mesh + a CT volume into a surface-volume stack** (what `vc_render` / `vc_render_tifxyz` do today). I'd like feedback from people who use these tools daily before I put serious effort into benchmarking.

### Why

Rendering surface volumes is conceptually simple: sample the CT along each mesh normal and write N layers. In practice it involves multi-TB OME-Zarr volumes, memory pressure, and a fairly heavy C++/Qt build. I wanted to see whether a self-contained Rust binary (`cargo build --release`) with strictly bounded memory could be a useful complement, especially on Windows and on cloud workers.

### What's in the repo

- **`volumen-zarr`**: OME-Zarr reader with a thread-safe, size-bounded LRU chunk cache (`--max-cache-gb`), Blosc decompression, and an HTTP fetcher that maps `s3://vesuvius-challenge-open-data/...` to public HTTPS.
- **`volumen-geom`**: Wavefront `.obj` and VC3D `tifxyz` loading, vertex normals, a UV spatial index for barycentric lookup, and 4×4 affine transforms (normals via (A⁻¹)ᵀ).
- **`volumen-sampler`**: trilinear sampling along normals, parallelized with Rayon, writing 16-bit TIFF stacks or OME-Zarr.
- **`volumen-cli`**: `volumen render | inspect | verify | unroll-spiral`, plus a `vc_render`-compatible binary target that accepts the familiar `-s/-v/-o/-n` syntax.
- **`volumen-py`**: PyO3 bindings returning NumPy arrays directly (`render_to_numpy`, `unroll_spiral_to_numpy`), for feeding ink-detection models without intermediate files.
- **Experimental**: `unroll-spiral` samples along an analytic Archimedean spiral, with an optional displacement field, as a way to prototype whole-scroll unrolling ideas.

Dual MIT/Apache-2.0 licensed.

### What I've actually validated so far (honestly, not much yet)

- **One real-data check**: on Scroll 1 (`54keV_7.91um_Scroll1A`), I used a small public segment (`20230503225234`, 100 vertices / 157 faces) and its official layer `32.tif` (249 × 85 px).
  - My barycentric sampling **prototype (Python)** matches the official `vc_render` output with **Pearson r = 0.9948** over 16,387 valid pixels.
  - Raw MAE is 10.6% of the 8-bit range. That drops to 0.61% after fitting a linear intensity transfer (y = 0.568x + 92), which I believe reflects the windowing of the reference TIFF, though I haven't confirmed that.
  - This validates the *algorithm* on a tiny patch. It does **not** yet validate the Rust binary.
- **26 Rust unit and integration tests** across the workspace (core, geom, sampler, zarr, cache memory pressure, spiral unroll, differential testing, real scroll segment loading) all pass cleanly.
- **No speed comparison with `vc_render` has been done yet.** I'm deliberately not claiming any speedup until I've run both on the same segments and hardware.
- The spiral-unrolling comparison in the repo is on a **synthetic, perfectly Archimedean spiral**, where an analytic method wins by construction. Real scrolls are crushed and deformed, so treat that part as a sandbox.

### What it is NOT

- Not a segmenter or surface tracer, and not a replacement for VC3D, Khartes, ThaumatoAnakalyptor or the official spiral-fitting work.
- Not an ink-detection model.

### Feedback I'm looking for

- Which segments and settings would make a convincing differential test against `vc_render_tifxyz`? Is there a standard "golden" segment people use?
- How do you handle intensity windowing in the published `layers/*.tif`? Is my linear-transfer explanation right?
- Is a `vc_render`-compatible CLI actually useful, or would a Python-first API (render to NumPy/Zarr) matter more to you?
- Rust folks: is a bounded LRU over decompressed chunks the right design, or should I cache compressed chunks and decompress on demand?

Repo: https://github.com/silveremartin-dev/Volumen

Thanks. Criticism very welcome, especially from people who've fought with these volumes before.

— Silvère Martin-Michiellot
