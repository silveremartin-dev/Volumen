use std::fs::{create_dir_all, File};
use std::io::Write;
use std::path::{Path, PathBuf};
use volumen_core::Result;

/// High-performance writer for chunked OME-Zarr arrays.
pub struct ZarrWriter {
    pub root_dir: PathBuf,
    pub shape: [u64; 3],  // [dim_z, dim_y, dim_x]
    pub chunks: [u64; 3], // [chunk_z, chunk_y, chunk_x]
}

impl ZarrWriter {
    /// Initializes a new Zarr v2 array directory on disk with `.zarray` metadata.
    pub fn create<P: AsRef<Path>>(dir: P, shape: [u64; 3], chunks: [u64; 3]) -> Result<Self> {
        let root = dir.as_ref().to_path_buf();
        create_dir_all(&root)?;

        let metadata_json = serde_json::json!({
            "zarr_format": 2,
            "shape": shape,
            "chunks": chunks,
            "dtype": "<u2",
            "order": "C",
            "compressor": null,
            "fill_value": 0
        });

        let meta_path = root.join(".zarray");
        let mut file = File::create(&meta_path)?;
        file.write_all(serde_json::to_string_pretty(&metadata_json)?.as_bytes())?;

        Ok(Self {
            root_dir: root,
            shape,
            chunks,
        })
    }

    /// Writes an individual chunk to disk following Zarr standard path conventions (cz/cy/cx).
    pub fn write_chunk(&self, cz: u64, cy: u64, cx: u64, data: &[u16]) -> Result<()> {
        let chunk_dir = self.root_dir.join(format!("{}/{}", cz, cy));
        create_dir_all(&chunk_dir)?;
        let chunk_file = chunk_dir.join(format!("{}", cx));

        let mut file = File::create(chunk_file)?;
        // Convert u16 slice to little-endian bytes
        let mut byte_buffer = Vec::with_capacity(data.len() * 2);
        for &val in data {
            byte_buffer.extend_from_slice(&val.to_le_bytes());
        }
        file.write_all(&byte_buffer)?;
        Ok(())
    }

    /// Writes all layers of an extracted surface volume directly into chunked Zarr format.
    /// Slices are provided as a slice of 2D buffers (each of size `width * height`).
    pub fn write_slices(&self, slices: &[Vec<u16>], width: usize, height: usize) -> Result<()> {
        let num_layers = slices.len();
        if num_layers == 0 {
            return Ok(());
        }

        let cz_size = self.chunks[0] as usize;
        let cy_size = self.chunks[1] as usize;
        let cx_size = self.chunks[2] as usize;

        let num_cz = num_layers.div_ceil(cz_size);
        let num_cy = height.div_ceil(cy_size);
        let num_cx = width.div_ceil(cx_size);

        for cz in 0..num_cz {
            let z_start = cz * cz_size;
            let z_end = (z_start + cz_size).min(num_layers);

            for cy in 0..num_cy {
                let y_start = cy * cy_size;
                let y_end = (y_start + cy_size).min(height);

                for cx in 0..num_cx {
                    let x_start = cx * cx_size;
                    let x_end = (x_start + cx_size).min(width);

                    let mut chunk_buffer = vec![0u16; cz_size * cy_size * cx_size];

                    for (z_local, slice) in slices[z_start..z_end].iter().enumerate() {
                        for y in y_start..y_end {
                            let y_local = y - y_start;
                            let row_offset = y * width;

                            for x in x_start..x_end {
                                let x_local = x - x_start;
                                let chunk_idx =
                                    z_local * (cy_size * cx_size) + y_local * cx_size + x_local;
                                chunk_buffer[chunk_idx] = slice[row_offset + x];
                            }
                        }
                    }

                    self.write_chunk(cz as u64, cy as u64, cx as u64, &chunk_buffer)?;
                }
            }
        }

        tracing::info!(
            "Successfully wrote {} layers ({}x{}) to chunked Zarr at {}",
            num_layers,
            width,
            height,
            self.root_dir.display()
        );

        Ok(())
    }
}
