use image_classification::datasets::*;

fn main() {
    let dataset = Cifar100::new("./datasets", DatasetSplit::Test)
        .unwrap()
        .unwrap();
    for i in 0..10 {
        let (img, lbl) = &dataset[i];
        img.save(std::format!("img-{i}-{}.png", dataset.label_name(*lbl)))
            .unwrap();
    }
}
