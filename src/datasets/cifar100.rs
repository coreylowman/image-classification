use std::{
    fs::File,
    io::{BufReader, Read},
    path::Path,
};

use curl::easy::Easy;

use image::{Rgb, RgbImage};

use super::{
    errors::DownloadError,
    split::{DatasetSplit, SplitNotFoundError},
};

pub struct Cifar100 {
    data: Vec<(RgbImage, u8)>,
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

impl std::ops::Index<usize> for Cifar100 {
    type Output = (RgbImage, u8);
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

impl Cifar100 {
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

impl Cifar100 {
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
        let root = if root.ends_with("cifar100") {
            root.to_path_buf()
        } else {
            root.join("cifar100")
        };

        let data_dir = root.join(DIR_NAME);
        if !data_dir.exists() {
            download_all(&root)?;
        }
        let mut data = Vec::new();
        for &f in files {
            let f_path = data_dir.join(f);
            if !f_path.exists() {
                download_all(&root)?;
            }
            load_bin(f_path, &mut data)?;
        }

        Ok(Self { data })
    }
}

const URL: &str = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz";
const MD5: &str = "03b5dce01913d631647c71ecec9e9cb8";

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

#[rustfmt::skip]
pub const LABEL_NAMES: [&str; 100] = [
    // aquatic mammals
    "beaver", "dolphin", "otter", "seal", "whale",
    // fish
    "aquarium fish", "flatfish", "ray", "shark", "trout",
    // flowers
    "orchids", "poppies", "roses", "sunflowers", "tulips",
    // food containers
    "bottles", "bowls", "cans", "cups", "plates",
    // fruit and vegetables
    "apples", "mushrooms", "oranges", "pears", "sweet peppers",
    // household electrical devices
    "clock", "computer keyboard", "lamp", "telephone", "television",
    // household furniture
    "bed", "chair", "couch", "table", "wardrobe",
    // insects
    "bee", "beetle", "butterfly", "caterpillar", "cockroach",
    // large carnivores
    "bear", "leopard", "lion", "tiger", "wolf",
    // large man-made outdoor things
    "bridge", "castle", "house", "road", "skyscraper",
    // large natural outdoor scenes
    "cloud", "forest", "mountain", "plain", "sea",
    // large omnivores and herbivores
    "camel", "cattle", "chimpanzee", "elephant", "kangaroo",
    // medium-sized mammals
    "fox", "porcupine", "possum", "raccoon", "skunk",
    // non-insect invertebrates
    "crab", "lobster", "snail", "spider", "worm",
    // people
    "baby", "boy", "girl", "man", "woman",
    // reptiles
    "crocodile", "dinosaur", "lizard", "snake", "turtle",
    // small mammals
    "hamster", "mouse", "rabbit", "shrew", "squirrel",
    // trees
    "maple", "oak", "palm", "pine", "willow",
    // vehicles 1
    "bicycle", "bus", "motorcycle", "pickup truck", "train",
    // vehicles 2
    "lawn-mower", "rocket", "streetcar", "tank", "tractor",
];
