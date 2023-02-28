use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use image::{Rgb, RgbImage};

use super::{
    download::{download_to, DownloadError},
    split::{Test, Train},
};

pub struct Cifar10<S> {
    data: Vec<(RgbImage, usize)>,
    pub split: S,
}

impl<S> std::ops::Index<usize> for Cifar10<S> {
    type Output = (RgbImage, usize);
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl<S> dfdx::data::ExactSizeDataset for Cifar10<S> {
    type Item<'a> = &'a (image::RgbImage, usize) where Self: 'a;
    fn get(&self, index: usize) -> Self::Item<'_> {
        &self.data[index]
    }
    fn len(&self) -> usize {
        self.data.len()
    }
}

impl<S> Cifar10<S> {
    pub fn label_name(&self, lbl: u8) -> &'static str {
        LABEL_NAMES[lbl as usize]
    }
}

impl Cifar10<Train> {
    pub fn new<P: AsRef<Path>>(root: P) -> Result<Self, DownloadError> {
        Self::load(root, Train, &TRAIN_FILES)
    }
}

impl Cifar10<Test> {
    pub fn new<P: AsRef<Path>>(root: P) -> Result<Self, DownloadError> {
        Self::load(root, Test, &TEST_FILES)
    }
}

impl<S> Cifar10<S> {
    fn load<P: AsRef<Path>>(root: P, split: S, files: &[&str]) -> Result<Self, DownloadError> {
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

        Ok(Self { data, split })
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

fn load_bin<P: AsRef<Path>>(
    path: P,
    data: &mut Vec<(RgbImage, usize)>,
) -> Result<(), std::io::Error> {
    let f = File::open(path)?;
    assert_eq!(f.metadata()?.len(), 3073 * 10_000);

    let mut r = BufReader::new(f);
    for _ in 0..10_000 {
        let mut lbl_buf = [0u8; 1];
        r.read_exact(&mut lbl_buf)?;
        let lbl = lbl_buf[0] as usize;

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
