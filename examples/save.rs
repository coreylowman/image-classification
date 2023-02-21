use image_classification::datasets::{cifar10::Cifar10, split::DatasetSplit};

fn main() {
    let dataset = Cifar10::new("./datasets", DatasetSplit::Test)
        .unwrap()
        .unwrap();
    for i in 0..10 {
        let (img, lbl) = &dataset[i];
        img.save(std::format!("img-{i}-{lbl:?}.png")).unwrap();
    }
}
