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

pub struct Cifar100 {
    data: Vec<(RgbImage, u8)>,
}

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

    pub fn label_name(&self, lbl: u8) -> &'static str {
        LABEL_NAMES[lbl as usize]
    }
}

impl Cifar100 {
    pub fn new<P: AsRef<Path>>(
        root: P,
        split: DatasetSplit,
    ) -> Result<Result<Self, DownloadError>, SplitNotFoundError> {
        match split {
            DatasetSplit::Train => Ok(Self::load(root, TRAIN_FILE, NUM_TRAIN_EXAMPLES)),
            DatasetSplit::Test => Ok(Self::load(root, TEST_FILE, NUM_TEST_EXAMPLES)),
            DatasetSplit::Val => Err(SplitNotFoundError(split)),
        }
    }

    fn load<P: AsRef<Path>>(root: P, file: &str, num: usize) -> Result<Self, DownloadError> {
        let root = root.as_ref();
        let root = if root.ends_with("cifar100") {
            root.to_path_buf()
        } else {
            root.join("cifar100")
        };

        if !root.exists() || !root.join(file).exists() {
            let uncompressed = download_to(&root, URL, MD5)?;
            let mut archive = tar::Archive::new(&uncompressed[..]);
            archive.unpack(&root)?;
        }
        let mut data = Vec::new();
        load_bin(&root.join(file), &mut data, num)?;

        Ok(Self { data })
    }
}

const URL: &str = "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz";
const MD5: &str = "03b5dce01913d631647c71ecec9e9cb8";

const NUM_TRAIN_EXAMPLES: usize = 50_000;
const NUM_TEST_EXAMPLES: usize = 10_000;
const TRAIN_FILE: &str = "cifar-100-binary/train.bin";
const TEST_FILE: &str = "cifar-100-binary/test.bin";

fn load_bin<P: AsRef<Path>>(
    path: P,
    data: &mut Vec<(RgbImage, u8)>,
    num: usize,
) -> Result<(), std::io::Error> {
    let f = File::open(path)?;
    assert_eq!(f.metadata()?.len(), 3074 * num as u64);

    let mut r = BufReader::new(f);
    for _ in 0..num {
        let mut lbl_buf = [0u8; 2];
        r.read_exact(&mut lbl_buf)?;
        let lbl = lbl_buf[1]; // use fine grained label

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

#[rustfmt::skip]
pub const LABEL_NAMES: [&str; 100] = [
    "apple","aquatic_fish","baby","bear","beaver",
    "bed", "bee", "beetle", "bicycle", "bottle",
    "bowl", "boy", "bridge", "bus", "butterfly",
    "camel", "can", "castle", "caterpillar", "cattle",
    "chair", "chimpanzee", "clock", "cloud", "cockroach",
    "couch", "crab", "crocodile", "cup", "dinosaur",
    "dolphin", "elephant", "flatfish", "forest", "fox",
    "girl", "hamster", "house", "kangaroo", "keyboard",
    "lamp", "lawn_mower", "leopard", "lion", "lizard",
    "lobster", "man", "maple_tree", "motorcycle", "mountain",
    "mouse", "mushroom", "oak_tree", "orange", "orchid",
    "otter", "palm_tree", "pear", "pickup_truck", "pine_tree",
    "plain", "plate", "poppy", "porcupine", "possum",
    "rabbit", "raccoon", "ray", "road", "rocket",
    "rose", "sea", "seal", "shark", "shrew",
    "skunk", "skyscraper", "snail", "snake", "spider",
    "squirrel", "streetcar", "sunflower", "sweet_pepper", "table",
    "tank", "telephone", "television", "tiger", "tractor",
    "train", "trout", "tulip", "turtle", "wardrobe",
    "whale", "willow_tree", "wolf", "woman", "worm",
];
