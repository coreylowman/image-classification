pub mod cifar10;
pub mod errors;
pub mod mnist;
pub mod split;

pub use cifar10::Cifar10;
pub use mnist::Mnist;
pub use split::DatasetSplit;
