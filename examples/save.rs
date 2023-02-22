use image_classification::datasets::{Cifar10, DatasetSplit, Mnist};

fn main() {
    let dataset = Mnist::new("./datasets", DatasetSplit::Test)
        .unwrap()
        .unwrap();
    for i in 0..10 {
        let (img, lbl) = &dataset[i];
        img.save(std::format!("img-{i}-{lbl:?}.png")).unwrap();
    }
}
