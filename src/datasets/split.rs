#[derive(Debug, Clone, Copy)]
pub enum DatasetSplit {
    Train,
    Test,
    Val,
}

#[derive(Debug)]
pub struct SplitNotFoundError(pub DatasetSplit);
