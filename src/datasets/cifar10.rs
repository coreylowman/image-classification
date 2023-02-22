use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use image::{Rgb, RgbImage};

use super::{
    download::{download_to, DownloadError},
    split::{DatasetSplit, SplitNotFoundError},
};

pub struct Cifar10 {
    data: Vec<(RgbImage, u8)>,
}

impl std::ops::Index<usize> for Cifar10 {
    type Output = (RgbImage, u8);
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl dfdx::data::ExactSizeDataset for Cifar10 {
    type Item = (image::RgbImage, usize);
    fn get(&self, index: usize) -> Self::Item {
        let (img, lbl) = &self.data[index];
        (img.clone(), *lbl as usize)
    }
    fn len(&self) -> usize {
        self.data.len()
    }
}

impl Cifar10 {
    pub fn label_name(&self, lbl: u8) -> &'static str {
        LABEL_NAMES[lbl as usize]
    }
}

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
        let root = if root.ends_with("cifar10") {
            root.to_path_buf()
        } else {
            root.join("cifar10")
        };

        if !root.exists() || files.iter().any(|f| !root.join(f).exists()) {
            let uncompressed = download_to(&root, URL, MD5)?;
            let mut archive = tar::Archive::new(&uncompressed[..]);
            archive.unpack(&root)?;
        }
        let mut data = Vec::new();
        for &f in files {
            load_bin(&root.join(f), &mut data)?;
        }

        Ok(Self { data })
    }
}

const URL: &str = "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz";
const MD5: &str = "c32a1d4ab5d03f1284b67883e8d87530";
const TRAIN_FILES: [&str; 5] = [
    "cifar-10-batches-bin/data_batch_1.bin",
    "cifar-10-batches-bin/data_batch_2.bin",
    "cifar-10-batches-bin/data_batch_3.bin",
    "cifar-10-batches-bin/data_batch_4.bin",
    "cifar-10-batches-bin/data_batch_5.bin",
];
const TEST_FILES: [&str; 1] = ["cifar-10-batches-bin/test_batch.bin"];

fn load_bin<P: AsRef<Path>>(path: P, data: &mut Vec<(RgbImage, u8)>) -> Result<(), std::io::Error> {
    let f = File::open(path)?;
    assert_eq!(f.metadata()?.len(), 3073 * 10_000);

    let mut r = BufReader::new(f);
    for _ in 0..10_000 {
        let mut lbl_buf = [0u8; 1];
        r.read_exact(&mut lbl_buf)?;
        let lbl = lbl_buf[0];

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

pub const LABEL_NAMES: [&str; 10] = [
    "Airplane",
    "Automobile",
    "Bird",
    "Cat",
    "Deer",
    "Dog",
    "Frog",
    "Horse",
    "Ship",
    "Truck",
];
