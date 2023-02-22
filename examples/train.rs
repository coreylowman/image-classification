use dfdx::{data::*, optim::Sgd, prelude::*};
use image_classification::datasets::{Cifar10, DatasetSplit};
use indicatif::ProgressIterator;
use rand::prelude::*;

type ResidualBlock<const C: usize, const D: usize> = (
    (Conv2D<C, D, 3, 1, 1>, MaxPool2D<3>, BatchNorm2D<D>, ReLU),
    Residual<(Conv2D<D, D, 3, 1, 1>, BatchNorm2D<D>, ReLU)>,
);

type SmallResnet<const NUM_CLASSES: usize> = (
    (Conv2D<3, 32, 3>, BatchNorm2D<32>, ReLU, MaxPool2D<3>),
    ResidualBlock<32, 64>,
    ResidualBlock<64, 128>,
    ResidualBlock<128, 256>,
    (AvgPoolGlobal, Linear<256, NUM_CLASSES>),
);

fn main() {
    let dev: Cpu = Default::default();
    let mut rng = StdRng::seed_from_u64(0);

    let mut model = dev.build_module::<SmallResnet<10>, f32>();
    let mut opt = Sgd::new(&model, Default::default());

    let train_data = Cifar10::new("./datasets", DatasetSplit::Train)
        .unwrap()
        .unwrap();

    let batch_size = Const::<4>;

    for i_epoch in 0.. {
        for (img, lbl) in train_data
            .shuffled(&mut rng)
            .progress()
            .batch(batch_size)
            .collate()
        {
            let imgs = dev.stack(img.map(|i| {
                dev.tensor((
                    i.iter().map(|&p| p as f32 / 255.0).collect(),
                    (Const::<3>, Const::<32>, Const::<32>),
                ))
            }));
            let lbls = dev.one_hot_encode(Const::<10>, lbl);

            let logits = model.forward_mut(imgs.traced());
            let loss = cross_entropy_with_logits_loss(logits, lbls);
            let grads = loss.backward();
            opt.update(&mut model, grads).unwrap();
        }
    }
}
