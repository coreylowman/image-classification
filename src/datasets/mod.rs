pub mod cifar10;
pub mod cifar100;
pub(crate) mod download;
pub mod mnist;
pub mod split;

pub use cifar10::Cifar10;
pub use cifar100::Cifar100;
pub use mnist::Mnist;
pub use split::DatasetSplit;
