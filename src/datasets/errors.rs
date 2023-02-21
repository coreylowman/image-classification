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

#[derive(Debug, Clone, Copy)]
pub struct LabelOrdinalError {
    pub found: usize,
    pub max: usize,
}
