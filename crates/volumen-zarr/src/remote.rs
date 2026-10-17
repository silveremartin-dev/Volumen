use crate::metadata::ZarrMetadata;
use std::process::Command;
use volumen_core::{Result, VolumenError};

/// Remote HTTP/S3 stream fetcher for out-of-core OME-Zarr arrays.
pub struct RemoteZarrFetcher;

impl RemoteZarrFetcher {
    /// Translates S3 protocol URIs (e.g. `s3://vesuvius-challenge-open-data/path`)
    /// to standard public HTTPS REST endpoints.
    pub fn resolve_url(path_or_url: &str) -> String {
        let trimmed = path_or_url.trim_end_matches('/');
        if let Some(s3_path) = trimmed.strip_prefix("s3://") {
            if let Some((bucket, key)) = s3_path.split_once('/') {
                format!("https://{}.s3.amazonaws.com/{}", bucket, key)
            } else {
                format!("https://{}.s3.amazonaws.com", s3_path)
            }
        } else {
            trimmed.to_string()
        }
    }

    /// Fetches `.zarray` metadata from remote HTTP/S3 URL.
    pub fn fetch_metadata(base_url: &str) -> Result<ZarrMetadata> {
        let resolved = Self::resolve_url(base_url);
        let meta_url = format!("{}/.zarray", resolved);

        tracing::info!("Fetching remote OME-Zarr metadata from {}", meta_url);
        let bytes = Self::fetch_bytes(&meta_url)?.ok_or_else(|| {
            VolumenError::Zarr(format!(".zarray not found at remote URL: {}", meta_url))
        })?;

        let json_str = std::str::from_utf8(&bytes)
            .map_err(|e| VolumenError::Zarr(format!("Invalid UTF-8 in remote .zarray: {}", e)))?;

        let meta: ZarrMetadata = serde_json::from_str(json_str).map_err(|e| {
            VolumenError::Zarr(format!("Failed to parse remote .zarray JSON: {}", e))
        })?;

        Ok(meta)
    }

    /// Fetches raw bytes from a remote URL using standard system HTTP transport.
    /// Returns `Ok(None)` if the chunk returns 404 (standard Zarr empty background block).
    pub fn fetch_bytes(url: &str) -> Result<Option<Vec<u8>>> {
        let output = Command::new("curl")
            .arg("-s") // Silent
            .arg("-L") // Follow redirects
            .arg("-f") // Fail silently on HTTP 4xx/5xx
            .arg("--retry") // Retry on network glitch
            .arg("2")
            .arg(url)
            .output();

        match output {
            Ok(out) => {
                if out.status.success() {
                    Ok(Some(out.stdout))
                } else {
                    // Check if 404 or empty chunk
                    tracing::debug!(
                        "Remote chunk fetch returned status {} for {}",
                        out.status,
                        url
                    );
                    Ok(None)
                }
            }
            Err(e) => Err(VolumenError::Io(e)),
        }
    }
}
