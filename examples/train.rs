use std::time::Instant;

use dfdx::{data::*, optim::Sgd, prelude::*};
use image_classification::datasets::{Cifar10, Test, Train};
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

type Dev = Cuda;
type Dtype = f32;

fn main() {
    let dev: Dev = Default::default();
    let mut rng = StdRng::seed_from_u64(0);

    let mut model = dev.build_module::<SmallResnet<10>, Dtype>();
    let mut grads = model.alloc_grads();
    let mut opt = Sgd::new(&model, Default::default());

    let train_data = Cifar10::<Train>::new("./datasets").unwrap();

    let batch = Const::<128>;

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

    for i_epoch in 0..10 {
        for (img, lbl) in train_data
            .shuffled(&mut rng)
            .map(preprocess)
            .batch(batch)
            .collate()
            .stack()
            .take(128)
        {
            let start = Instant::now();
            let logits = model.forward_mut(img.traced(grads));
            let loss = cross_entropy_with_logits_loss(logits, lbl);
            dev.synchronize().unwrap();
            let fwd_dur = start.elapsed();
            let loss_val = loss.array();

            let start = Instant::now();
            grads = loss.backward();
            dev.synchronize().unwrap();
            let bwd_dur = start.elapsed();

            let start = Instant::now();
            opt.update(&mut model, &grads).unwrap();
            model.zero_grads(&mut grads);
            dev.synchronize().unwrap();
            let opt_dur = start.elapsed();

            println!(
                "loss={loss_val} | fwd={:?} bwd={:?} opt={:?}",
                fwd_dur, bwd_dur, opt_dur
            );
        }
    }

    // Create the test dataset
    let test_data = {
        let mut data = Vec::with_capacity(10_000);
        for (img, lbl) in Cifar10::<Test>::new("./datasets").unwrap().iter() {
            let mut one_hotted = [0.0; 10];
            one_hotted[*lbl] = 1.0;
            data.push((
                dev.tensor_from_vec(
                    img.iter().map(|&p| p as Dtype / 255.0).collect(),
                    (Const::<3>, Const::<32>, Const::<32>),
                ),
                dev.tensor(one_hotted),
            ));
        }
        data
    };

    // Create a counter value
    let mut correct = 0;
    // Evaluate each image in the test set using the trained model
    for i in 0..test_data.len() {
        // Get the i'th image from the test dataset
        let img = test_data[i].0.clone();
        // Evaluate it via forwarding with the model
        let base_result = model.forward(img);
        // Convert the result from a tensor to a stdlib Vec
        let result = dev.tensor_to_vec(&base_result);

        // Get the i'th label from the test dataset and select the values
        let label = &test_data[i].1;
        // Convert those labels to a Vec<Dtype> (Vec<f32>)
        let label: Vec<Dtype> = dev
            .tensor_to_vec(label)
            .iter()
            .map(|x| *x as Dtype)
            .collect();

        // We could print the outputs side-to-side for each label vs. evaluated result
        // println!("{:.2?} | {:?}", &result, &label);

        // Compare the labels, and if the max index of each vector is the same,
        // count that as correct and increment the counter
        if (get_max_index(result) == get_max_index(label)) {
            correct += 1;
        }
    }
    // Print the total correct
    dbg!(correct);
}

fn get_max_index(v: Vec<f32>) -> usize {
    let mut max_index = 0;
    for i in 0..v.len() {
        if v[i] > v[max_index] {
            max_index = i;
        }
    }
    max_index
}
