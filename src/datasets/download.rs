use std::{io::Read, path::Path};

use curl::easy::Easy;

pub(crate) fn download_to<P: AsRef<Path>>(
    root: P,
    url: &str,
    md5: &str,
) -> Result<Vec<u8>, DownloadError> {
    let root = root.as_ref();
    std::fs::create_dir_all(root)?;

    let mut compressed = Vec::new();
    let mut easy = Easy::new();
    easy.url(url).unwrap();
    easy.progress(true).unwrap();

    println!("Downloading {url}");
    {
        let mut dl = easy.transfer();
        let pb = indicatif::ProgressBar::new(1);
        dl.progress_function(move |total_dl, cur_dl, _, _| {
            pb.set_length(total_dl as u64);
            pb.set_position(cur_dl as u64);
            true
        })?;
        dl.write_function(|data| {
            compressed.extend_from_slice(data);
            Ok(data.len())
        })?;
        dl.perform()?;
    }

    println!("Verifying hash is {md5}");
    let digest = md5::compute(&compressed);
    if format!("{:?}", digest) != md5 {
        return Err(DownloadError::Md5Mismatch);
    }

    println!("Deflating {} bytes", compressed.len());
    let mut uncompressed = Vec::new();
    let mut decoder = flate2::read::GzDecoder::new(&compressed[..]);
    decoder.read_to_end(&mut uncompressed)?;
    Ok(uncompressed)
}

#[derive(Debug)]
pub enum DownloadError {
    IoError(std::io::Error),
    CurlError(curl::Error),
    Md5Mismatch,
}

impl std::fmt::Display for DownloadError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_fmt(format_args!("{:?}", self))
    }
}

impl std::error::Error for DownloadError {}

impl From<std::io::Error> for DownloadError {
    fn from(e: std::io::Error) -> Self {
        Self::IoError(e)
    }
}

impl From<curl::Error> for DownloadError {
    fn from(e: curl::Error) -> Self {
        Self::CurlError(e)
    }
}
