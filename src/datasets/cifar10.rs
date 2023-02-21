use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use curl::easy::Easy;

use image::{Rgb, RgbImage};

use super::{
    errors::{DownloadError, LabelOrdinalError},
    split::{DatasetSplit, SplitNotFoundError},
};

pub struct Cifar10 {
    data: Vec<(RgbImage, Label)>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Label {
    Airplane,
    Automobile,
    Bird,
    Cat,
    Deer,
    Dog,
    Frog,
    Horse,
    Ship,
    Truck,
}

impl From<Label> for usize {
    fn from(value: Label) -> Self {
        match value {
            Label::Airplane => 0,
            Label::Automobile => 1,
            Label::Bird => 2,
            Label::Cat => 3,
            Label::Deer => 4,
            Label::Dog => 5,
            Label::Frog => 6,
            Label::Horse => 7,
            Label::Ship => 8,
            Label::Truck => 9,
        }
    }
}

impl TryFrom<usize> for Label {
    type Error = LabelOrdinalError;
    fn try_from(value: usize) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::Airplane),
            1 => Ok(Self::Automobile),
            2 => Ok(Self::Bird),
            3 => Ok(Self::Cat),
            4 => Ok(Self::Deer),
            5 => Ok(Self::Dog),
            6 => Ok(Self::Frog),
            7 => Ok(Self::Horse),
            8 => Ok(Self::Ship),
            9 => Ok(Self::Truck),
            _ => Err(LabelOrdinalError {
                found: value,
                max: 9,
            }),
        }
    }
}

impl std::ops::Index<usize> for Cifar10 {
    type Output = (RgbImage, Label);
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl Cifar10 {
    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

const DIR_NAME: &str = "cifar-10-batches-bin";
const TRAIN_FILES: [&str; 5] = [
    "data_batch_1.bin",
    "data_batch_2.bin",
    "data_batch_3.bin",
    "data_batch_4.bin",
    "data_batch_5.bin",
];

const TEST_FILES: [&str; 1] = ["test_batch.bin"];

impl Cifar10 {
    pub fn new<P: AsRef<Path>>(
        root: P,
        split: DatasetSplit,
    ) -> Result<Result<Self, DownloadError>, SplitNotFoundError> {
        match split {
            DatasetSplit::Train => Ok(Self::load(root, &TRAIN_FILES)),
            DatasetSplit::Test => Ok(Self::load(root, &TEST_FILES)),
            DatasetSplit::Val => Err(SplitNotFoundError(split)),
        }
    }

    fn load<P: AsRef<Path>>(root: P, files: &[&str]) -> Result<Self, DownloadError> {
        let root = root.as_ref();

        let data_dir = root.join(DIR_NAME);
        if !data_dir.exists() {
            download_all(root)?;
        }
        let mut data = Vec::new();
        for &f in files {
            let f_path = data_dir.join(f);
            if !f_path.exists() {
                download_all(root)?;
            }
            load_bin(f_path, &mut data)?;
        }

        Ok(Self { data })
    }
}

const URL: &str = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
const MD5: &str = "c32a1d4ab5d03f1284b67883e8d87530";

fn load_bin<P: AsRef<Path>>(
    path: P,
    data: &mut Vec<(RgbImage, Label)>,
) -> Result<(), std::io::Error> {
    let f = File::open(path)?;
    assert_eq!(f.metadata()?.len(), 3073 * 10_000);

    let mut r = BufReader::new(f);
    for _ in 0..10_000 {
        let mut lbl_buf = [0u8; 1];
        r.read_exact(&mut lbl_buf)?;
        let lbl: Label = (lbl_buf[0] as usize).try_into().unwrap();

        let mut img_buf = vec![0u8; 3072];
        r.read_exact(&mut img_buf)?;
        let img = RgbImage::from_fn(32, 32, |x, y| {
            let x = x as usize;
            let y = y as usize;
            Rgb([
                img_buf[y * 32 + x],
                img_buf[32 * 32 + y * 32 + x],
                img_buf[2 * 32 * 32 + y * 32 + x],
            ])
        });

        data.push((img, lbl));
    }
    Ok(())
}

fn download_all<P: AsRef<Path>>(root: P) -> Result<(), DownloadError> {
    download(root, URL, MD5)
}

fn download<P: AsRef<Path>>(root: P, url: &str, md5: &str) -> Result<(), DownloadError> {
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

    let mut archive = tar::Archive::new(&uncompressed[..]);
    archive.unpack(root)?;

    Ok(())
}
