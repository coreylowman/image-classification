use image::GrayImage;
use std::{
    fs::File,
    io::{BufReader, BufWriter, Read, Write},
    path::Path,
};

use super::{
    download::{download_to, DownloadError},
    split::{Test, Train},
};

pub struct Mnist<Split> {
    data: Vec<(image::GrayImage, u8)>,
    pub split: Split,
}

impl Mnist<Train> {
    pub fn new<P: AsRef<Path>>(root: P) -> Result<Self, DownloadError> {
        Self::load(root, Train, TRAIN_IMG_NAME, TRAIN_LBL_NAME, 60_000)
    }
}

impl Mnist<Test> {
    pub fn new<P: AsRef<Path>>(root: P) -> Result<Self, DownloadError> {
        Self::load(root, Test, TEST_IMG_NAME, TEST_LBL_NAME, 10_000)
    }
}

impl<S> Mnist<S> {
    fn load<P: AsRef<Path>>(
        root: P,
        split: S,
        img_name: &str,
        lbl_name: &str,
        num: usize,
    ) -> Result<Self, DownloadError> {
        let root = root.as_ref();
        let root = if root.ends_with("mnist") {
            root.to_path_buf()
        } else {
            root.join("mnist")
        };

        let img_path = root.join(img_name);
        let lbl_path = root.join(lbl_name);
        if !img_path.exists() || !lbl_path.exists() {
            download_all(root)?;
        }

        let mut pixels = checked_open(img_path, &[num, 28, 28])?;
        let mut labels = checked_open(lbl_path, &[num])?;

        let mut data = Vec::new();
        for _ in 0..num {
            let mut img_buf = vec![0u8; 28 * 28];
            let mut lbl_buf = [0u8; 1];

            pixels.read_exact(&mut img_buf)?;
            labels.read_exact(&mut lbl_buf)?;

            let img = GrayImage::from_vec(28, 28, img_buf).unwrap();
            let lbl = lbl_buf[0];
            data.push((img, lbl))
        }
        Ok(Self { data, split })
    }
}

impl<S> Mnist<S> {
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

impl<S> std::ops::Index<usize> for Mnist<S> {
    type Output = (GrayImage, u8);
    fn index(&self, index: usize) -> &Self::Output {
        &self.data[index]
    }
}

fn checked_open<P: AsRef<Path>>(
    p: P,
    expected_shape: &[usize],
) -> Result<BufReader<File>, std::io::Error> {
    let f = File::open(p)?;
    let mut f = BufReader::new(f);

    let magic = read_u32(&mut f)?;
    let num_dims = magic % 256;

    let ty = magic / 256;
    assert_eq!(ty, 8); // u8

    let mut shape = Vec::with_capacity(num_dims as usize);
    for _ in 0..num_dims {
        let d = read_u32(&mut f)? as usize;
        shape.push(d);
    }
    assert_eq!(&shape, expected_shape);
    Ok(f)
}

fn read_u32<R: std::io::Read>(r: &mut R) -> Result<u32, std::io::Error> {
    let mut buf = [0; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_be_bytes(buf))
}

const BASE_URL: &str = "http://yann.lecun.com/exdb/mnist/";
const TRAIN_IMG_NAME: &str = "train-images-idx3-ubyte";
const TRAIN_LBL_NAME: &str = "train-labels-idx1-ubyte";
const TEST_IMG_NAME: &str = "t10k-images-idx3-ubyte";
const TEST_LBL_NAME: &str = "t10k-labels-idx1-ubyte";
const RESOURCES: [(&str, &str); 4] = [
    (
        "train-images-idx3-ubyte.gz",
        "f68b3c2dcbeaaa9fbdd348bbdeb94873",
    ),
    (
        "train-labels-idx1-ubyte.gz",
        "d53e105ee54ea40749a09fcbcd1e9432",
    ),
    (
        "t10k-images-idx3-ubyte.gz",
        "9fb629c4189551a2d022fa330f9573f3",
    ),
    (
        "t10k-labels-idx1-ubyte.gz",
        "ec29112dd5afa0611ce80d1b7f02629c",
    ),
];

fn download_all<P: AsRef<Path>>(root: P) -> Result<(), DownloadError> {
    let root = root.as_ref();
    for (name, md5) in RESOURCES {
        let uncompressed = download_to(root, &std::format!("{BASE_URL}{name}"), md5)?;
        let path = root.join(name.replace(".gz", ""));
        println!("Writing {} bytes to {}", uncompressed.len(), path.display());
        let mut o = BufWriter::new(File::create(path)?);
        o.write_all(&uncompressed)?;
    }
    Ok(())
}

const LABEL_NAMES: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];
