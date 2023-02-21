// use crate::prelude::{HasArrayData, SubsetIterator, Tensor2D, TensorCreator};
// use curl::easy::Easy;
// use std::{
//     fs::File,
//     io::{BufReader, BufWriter, Read, Write},
//     path::Path,
// };

// pub struct Mnist {
//     pixels: Vec<f32>,
//     labels: Vec<u8>,
// }

// const TRAIN_IMG_NAME: &str = "train-images-idx3-ubyte";
// const TRAIN_LBL_NAME: &str = "train-labels-idx1-ubyte";
// const TEST_IMG_NAME: &str = "t10k-images-idx3-ubyte";
// const TEST_LBL_NAME: &str = "t10k-labels-idx1-ubyte";

// impl Mnist {
//     pub fn train_data<P: AsRef<Path>>(root: P) -> Result<Self, MnistDownloadError> {
//         let root = root.as_ref();

//         let img_path = root.join(TRAIN_IMG_NAME);
//         let lbl_path = root.join(TRAIN_LBL_NAME);
//         if !img_path.exists() || !lbl_path.exists() {
//             download_all(root)?;
//         }

//         let pixels = load(img_path, &[60_000, 28, 28])?;
//         let labels = load(lbl_path, &[60_000])?;

//         Ok(Self {
//             pixels: pixels.iter().map(|&v| v as f32 / 255.0).collect(),
//             labels,
//         })
//     }

//     pub fn test_data<P: AsRef<Path>>(root: P) -> Result<Self, MnistDownloadError> {
//         let root = root.as_ref();

//         let img_path = root.join(TEST_IMG_NAME);
//         let lbl_path = root.join(TEST_LBL_NAME);
//         if !img_path.exists() || !lbl_path.exists() {
//             download_all(root)?;
//         }

//         let pixels = load(img_path, &[60_000, 28, 28])?;
//         let labels = load(lbl_path, &[60_000])?;

//         Ok(Self {
//             pixels: pixels.iter().map(|&v| v as f32 / 255.0).collect(),
//             labels,
//         })
//     }

//     pub fn is_empty(&self) -> bool {
//         self.labels.is_empty()
//     }

//     pub fn len(&self) -> usize {
//         self.labels.len()
//     }

//     pub fn get_batch<const B: usize>(
//         &self,
//         idxs: [usize; B],
//     ) -> (Tensor2D<B, { 1 * 28 * 28 }>, Tensor2D<B, 10>) {
//         let mut img = Tensor2D::zeros();
//         let mut lbl = Tensor2D::zeros();
//         let img_data = img.mut_data();
//         let lbl_data = lbl.mut_data();
//         for (batch_i, &img_idx) in idxs.iter().enumerate() {
//             let start = (1 * 28 * 28) * img_idx;
//             img_data[batch_i].copy_from_slice(&self.pixels[start..start + (1 * 28 * 28)]);
//             lbl_data[batch_i][self.labels[img_idx] as usize] = 1.0;
//         }
//         (img, lbl)
//     }

//     pub fn batches<R: rand::Rng, const B: usize>(
//         &self,
//         rng: &mut R,
//     ) -> impl '_ + Iterator<Item = (Tensor2D<B, { 1 * 28 * 28 }>, Tensor2D<B, 10>)> {
//         SubsetIterator::<B>::shuffled(self.len(), rng).map(|i| self.get_batch(i))
//     }
// }

// fn load<P: AsRef<Path>>(p: P, expected_shape: &[usize]) -> Result<Vec<u8>, std::io::Error> {
//     let f = File::open(p)?;
//     let mut f = BufReader::new(f);

//     let magic = read_u32(&mut f)?;
//     let num_dims = magic % 256;

//     let ty = magic / 256;
//     assert_eq!(ty, 8); // u8

//     let mut shape = Vec::with_capacity(num_dims as usize);
//     let mut num_elements = 1;
//     for _ in 0..num_dims {
//         let d = read_u32(&mut f)? as usize;
//         num_elements *= d;
//         shape.push(d);
//     }
//     assert_eq!(&shape, expected_shape);

//     let mut buf: Vec<u8> = Vec::with_capacity(num_elements);
//     f.read_to_end(&mut buf)?;
//     assert_eq!(buf.len(), num_elements);
//     Ok(buf)
// }

// fn read_u32<R: std::io::Read>(r: &mut R) -> Result<u32, std::io::Error> {
//     let mut buf = [0; 4];
//     r.read_exact(&mut buf)?;
//     Ok(u32::from_be_bytes(buf))
// }

// const BASE_URL: &str = "http://yann.lecun.com/exdb/mnist/";
// const RESOURCES: [(&str, &str); 4] = [
//     (
//         "train-images-idx3-ubyte.gz",
//         "f68b3c2dcbeaaa9fbdd348bbdeb94873",
//     ),
//     (
//         "train-labels-idx1-ubyte.gz",
//         "d53e105ee54ea40749a09fcbcd1e9432",
//     ),
//     (
//         "t10k-images-idx3-ubyte.gz",
//         "9fb629c4189551a2d022fa330f9573f3",
//     ),
//     (
//         "t10k-labels-idx1-ubyte.gz",
//         "ec29112dd5afa0611ce80d1b7f02629c",
//     ),
// ];

// fn download_all<P: AsRef<Path>>(root: P) -> Result<(), MnistDownloadError> {
//     let root = root.as_ref();
//     for (name, md5) in RESOURCES {
//         download(root, name, md5)?;
//     }
//     Ok(())
// }

// fn download<P: AsRef<Path>>(root: P, name: &str, md5: &str) -> Result<(), MnistDownloadError> {
//     let root = root.as_ref();
//     std::fs::create_dir_all(root)?;

//     let url = BASE_URL.to_owned() + name;

//     let mut compressed = Vec::new();
//     let mut easy = Easy::new();
//     easy.url(&url).unwrap();
//     easy.progress(true).unwrap();

//     println!("Downloading {name}");
//     {
//         let mut dl = easy.transfer();
//         let pb = indicatif::ProgressBar::new(1);
//         dl.progress_function(move |total_dl, cur_dl, _, _| {
//             pb.set_length(total_dl as u64);
//             pb.set_position(cur_dl as u64);
//             true
//         })?;
//         dl.write_function(|data| {
//             compressed.extend_from_slice(data);
//             Ok(data.len())
//         })?;
//         dl.perform()?;
//     }

//     println!("Verifying hash is {md5}");
//     let digest = md5::compute(&compressed);
//     if format!("{:?}", digest) != md5 {
//         return Err(MnistDownloadError::Md5Mismatch);
//     }

//     println!("Deflating {} bytes", compressed.len());
//     let mut uncompressed = Vec::new();
//     let mut decoder = flate2::read::GzDecoder::new(&compressed[..]);
//     decoder.read_to_end(&mut uncompressed)?;

//     let path = root.join(name.replace(".gz", ""));
//     println!("Writing {} bytes to {}", uncompressed.len(), path.display());
//     let mut o = BufWriter::new(File::create(path)?);
//     o.write_all(&uncompressed)?;

//     Ok(())
// }
