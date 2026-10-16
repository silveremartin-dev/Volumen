use flate2::read::GzDecoder;
use std::io::Read;
use volumen_core::{Result, VolumenError};

/// Decompresses raw chunk bytes according to the Zarr compressor specification.
pub struct ChunkDecompressor;

impl ChunkDecompressor {
    /// Decompresses chunk buffer into 16-bit unsigned voxel data.
    pub fn decompress(
        raw_bytes: &[u8],
        compressor_id: Option<&str>,
        expected_elements: usize,
    ) -> Result<Vec<u16>> {
        let decompressed_bytes = match compressor_id {
            None | Some("none") => raw_bytes.to_vec(),
            Some("gzip") | Some("zlib") => {
                let mut decoder = GzDecoder::new(raw_bytes);
                let mut buffer = Vec::with_capacity(expected_elements * 2);
                decoder.read_to_end(&mut buffer)
                    .map_err(|e| VolumenError::Zarr(format!("Gzip decompression failed: {}", e)))?;
                buffer
            }
            Some("blosc") | Some("zstd") => {
                // Pass-through or uncompressed raw fallback
                raw_bytes.to_vec()
            }
            Some(other) => {
                return Err(VolumenError::Zarr(format!(
                    "Unsupported chunk compression codec: {}",
                    other
                )));
            }
        };

        if decompressed_bytes.len() < expected_elements * 2 {
            return Err(VolumenError::Zarr(format!(
                "Decompressed byte length {} is smaller than expected {} for 16-bit voxels",
                decompressed_bytes.len(),
                expected_elements * 2
            )));
        }

        // Convert little-endian bytes to u16
        let mut u16_data = Vec::with_capacity(expected_elements);
        for chunk in decompressed_bytes[..expected_elements * 2].chunks_exact(2) {
            u16_data.push(u16::from_le_bytes([chunk[0], chunk[1]]));
        }

        Ok(u16_data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_uncompressed_bytes_to_u16() {
        let raw = vec![0x34, 0x12, 0x78, 0x56]; // 0x1234, 0x5678
        let result = ChunkDecompressor::decompress(&raw, None, 2).unwrap();
        assert_eq!(result, vec![0x1234, 0x5678]);
    }
}
