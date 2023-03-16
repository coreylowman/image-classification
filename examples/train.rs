use std::time::Instant;

use dfdx::{data::*, optim::Sgd, prelude::*};
use image_classification::datasets::{Cifar10, Train};
use rand::prelude::*;

type ResidualBlock<const C: usize, const D: usize> = (
    (Conv2D<C, D, 3, 1, 1>, BatchNorm2D<D>, MaxPool2D<3>, ReLU),
    Residual<(Conv2D<D, D, 3, 1, 1>, BatchNorm2D<D>, ReLU)>,
);

type SmallResnet<const NUM_CLASSES: usize> = (
    (Conv2D<3, 32, 3>, BatchNorm2D<32>, ReLU, MaxPool2D<3>),
    ResidualBlock<32, 64>,
    ResidualBlock<64, 128>,
    ResidualBlock<128, 256>,
    (AvgPoolGlobal, Linear<256, NUM_CLASSES>),
);

type Dev = Cpu;
type Dtype = f32;

fn main() {
    let dev: Dev = Default::default();
    let mut rng = StdRng::seed_from_u64(0);

    let mut model = dev.build_module::<SmallResnet<10>, Dtype>();
    let mut grads = model.alloc_grads();
    let mut opt = Sgd::new(&model, Default::default());

    let train_data = Cifar10::<Train>::new("./datasets").unwrap();

    let batch = Const::<16>;

    let preprocess = |(img, lbl): <Cifar10<Train> as ExactSizeDataset>::Item<'_>| {
        let mut one_hotted = [0.0; 10];
        one_hotted[*lbl] = 1.0;
        (
            dev.tensor_from_vec(
                img.iter().map(|&p| p as Dtype / 255.0).collect(),
                (Const::<3>, Const::<32>, Const::<32>),
            ),
            dev.tensor(one_hotted),
        )
    };

    for i_epoch in 0.. {
        for (img, lbl) in train_data
            .shuffled(&mut rng)
            .map(preprocess)
            .batch(batch)
            .collate()
            .stack()
        {
            let start = Instant::now();
            let logits = model.forward_mut(img.traced(grads));
            let loss = cross_entropy_with_logits_loss(logits, lbl);
            let fwd_dur = start.elapsed();
            let loss_val = loss.array();

            let start = Instant::now();
            grads = loss.backward();
            let bwd_dur = start.elapsed();

            let start = Instant::now();
            opt.update(&mut model, &grads).unwrap();
            model.zero_grads(&mut grads);
            let opt_dur = start.elapsed();

            println!(
                "loss={loss_val} | fwd={:?} bwd={:?} opt={:?}",
                fwd_dur, bwd_dur, opt_dur
            );
        }
    }
}
