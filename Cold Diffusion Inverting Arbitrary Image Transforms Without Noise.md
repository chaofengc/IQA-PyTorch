# Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise
<img src="./all_transform_cover.png" width="1000px"></img>

The official PyTorch implementation of <a href="https://arxiv.org/abs/2208.09392">Cold-Diffusion</a>. Developed collaboratively by Arpit Bansal, Eitan Borgnia, Hong-Min Chu, Jie Li, and Hamid Kazemi, all at the University of Maryland. This repository has code to train and test various cold diffusion models based on the following image degradations: Gaussian blur, _animorphosis_, Gaussian mask, resolution downsampling, image snow, and color desaturation. Our implementation is based on the denoising diffusion repository from <a href="https://github.com/lucidrains/denoising-diffusion-pytorch">lucidrains</a>, which is a PyTorch implementation of <a href="https://arxiv.org/abs/2006.11239">DDPM</a>.

## Updates
October 12, 2022 : Pretrained models for CelebA generation using blur and animorphs, and AFHQ generation using blur, added to <a href="https://drive.google.com/drive/folders/1R7CKUrkiIDsDYh2__Yi1iLvRR6wNxVFF?usp=sharing">our drive</a>.

## Citing Our Work

To cite our paper please use the following bibtex entry.

```bibtex
@misc{bansal2022cold,
      title={Cold Diffusion: Inverting Arbitrary Image Transforms Without Noise}, 
      author={Arpit Bansal and Eitan Borgnia and Hong-Min Chu and Jie S. Li and Hamid Kazemi and Furong Huang and Micah Goldblum and Jonas Geiping and Tom Goldstein},
      year={2022},
      eprint={2208.09392},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## Preparing the Datasets

We use the `create_data.py` file to split data into individual folders for training and testing data. The MNIST and CIFAR-10 datasets can be processed directly with `create_data.py`, but a path to the folder for the CelebA dataset is required. The AFHQ dataset is already split up into individual folders for training and testing data, so preprocessing is unecessary. The save directory for each of the data folders can be modified and will be used in the training scripts for the various hot/cold diffusion models.

## Denoising (hot) Diffusion Models

For completeness, we present the code to produce models trained to undo Gaussian noise for direct comparison to our cold diffusion models on the task of unconditional generation. The code for these models is in the `denoising-diffusion-pytorch` folder.

#### Training 

We have separate scripts for training models on each dataset _i.e_ `<dataset>_noise_<resolution>.py`. As per Section 5 of the paper, we include scripts for the 128 x 128 CelebA and AFHQ datasets. The `--time_steps` argument can used to vary the number of steps it takes to reach the final isotropic Gaussian noise distribution. 

The `--sampling_routine` argument allows you to switch between different sampling algorithms. Choosing `default` will sample using Algorithm 1 from our paper, `x0_step_down` is Algorithm 2 from our paper with fixed noise, and `ddim` is the version of Algorithm 2 where one uses estimated noise as in <a href="https://arxiv.org/abs/2010.02502">Song et al.</a>.

The `--save_folder` argument indicates the path to save the trained model, and the training data samples produced to keep track of progress. The frequency of saving and progress tracking can be modified in the `Trainer` class defined in `denoising_diffusion_pytorch.py`. The `data_path` argument specifies the path to the training data folder produced in the dataset preparation step.

Below is an example script for training denoising diffusion models.

```
python <dataset>_noise_<resolution>.py --time_steps 200 --sampling_routine x0_step_down --save_folder <Path to save model folder> --data_path <Path to train data folder>
```

#### Testing

To test the models, it is important to input the same number of time steps used during training. The sampling routine can be different, and additionally the `test_type` argument can be used to specify the type of testing to be done. See the `Trainer` class defined in `denoising_diffusion_pytorch.py` to understand the different testing methods. 

We present two example testing scripts below. The first script uses the DDIM sampling method with estimated noise, and the second script uses the fixed noise method we describe in Algorithm 2 of our paper.

```
python <dataset>_noise_<resolution>_test.py --time_steps 200 --sampling_routine ddim --save_folder <Path to save images> --load_path <Path to load model> --data_path <Path to data folder> --test_type test_sample_and_save_for_fid
python <dataset>_noise_<resolution>_test.py --time_steps 200 --sampling_routine x0_step_down --save_folder <Path to save images> --load_path <Path to load model> --data_path <Path to data folder> --test_type test_sample_and_save_for_fid
```

## Cold Diffusion Models

We present the training and testing procedure for the various cold diffusions explored in our paper. For each cold diffusion, there are example scripts for two different degradation schedules. The conditional generation schedule can be used to reproduce results from Section 4 of the paper, and the unconditional generation schedule can be used to reproduce section 5. Unless otherwise specified, the arguments and procedure work exactly the same as described in detail for the hot diffusion models.

### Gaussian Blur

The code for these models is in the `deblurring-diffusion-pytorch` folder. For Gaussian blurring, the arguments that specify the degradation schedule are `--time_steps`, `--blur_size`, `--blur_std`, and `blur_routine`. See the help strings in the training scripts for details on what these arguments specify.

#### Conditional Generation

*Training*
```
python mnist_train.py --time_steps 20 --blur_size 11 --blur_std 7.0 --blur_routine 'Constant' --sampling_routine x0_step_down --data_path <Path to data folder> --save_folder <Path to save model> 
python cifar10_train.py --time_steps 50 --blur_routine 'Special_6_routine' --sampling_routine x0_step_down --data_path <Path to data folder> --save_folder <Path to save model> 
python celebA_128.py --time_steps 200 --blur_size 15 --blur_std 0.01 --blur_routine Exponential_reflect --sampling_routine x0_step_down --data_path <Path to data folder> --save_folder <Path to save model> 
```
*Testing*
```
python mnist_test.py --time_steps 20 --blur_size 11 --blur_std 7.0 --blur_routine 'Constant' --sampling_routine 'x0_step_down' --save_folder <Path to save results> --data_path <Path to data folder> --test_type test_data
python cifar10_test.py --time_steps 50 --blur_routine 'Special_6_routine' --sampling_routine 'x0_step_down' --save_folder <Path to save results> --data_path <Path to data folder> --test_type test_data
python celebA_128_test.py --time_steps 200 --blur_size 15 --blur_std 0.01 --blur_routine Exponential_reflect --sampling_routine x0_step_down --save_folder <Path to save results> --data_path <Path to data folder> --test_type test_data
```

#### Unconditional Generation

*Training*
```
python celebA_128.py --discrete --time_steps 300 --blur_size 27 --blur_std 0.01 --blur_routine Exponential --sampling_routine x0_step_down --data_path <Path to data folder> --save_folder <Path to save models>
python AFHQ_128.py --discrete --time_steps 300 --blur_size 27 --blur_std 0.01 --blur_routine Exponential --sampling_routine x0_step_down --data_path <Path to data folder> --save_folder <Path to save models>
```
*Testing*

Below are two sets of testing scripts for the models/datasets presented in the paper. The first set of scripts corresponds to the sampling done with **perfect symmetry**, and the second set of scripts corresponds to sampling done with **broken** symmetry. The `--gmm_cluster` argument specifies the number of modes for the Gaussian mixture model (GMM) that is fit to the distribution of degraded images. The `--noise` argument is used to control the degree to which symmetry is broken after sampling from the GMM.
```
python celebA_128_test.py --gmm_cluster 1 --noise 0.000 --discrete --time_steps 300 --blur_size 27 --blur_std 0.01 --blur_routine Exponential --sampling_routine x0_step_down --save_folder <Path to save results> --load_path <Path to load models> --data_path <Path to data folder> --test_type train_distribution_mean_blur_torch_gmm_ablation
python AFHQ_128_test.py --gmm_cluster 1 --noise 0.000 --discrete --time_steps 300 --blur_size 27 --blur_std 0.01 --blur_routine Exponential --sampling_routine x0_step_down --save_folder <Path to save results> --load_path <Path to load models> --data_path <Path to data folder> --test_type train_distribution_mean_blur_torch_gmm_ablation
python celebA_128_test.py --gmm_cluster 1 --noise 0.002 --discrete --time_steps 300 --blur_size 27 --blur_std 0.01 --blur_routine Exponential --sampling_routine x0_step_down --save_folder <Path to save results> --load_path <Path to load models> --data_path <Path to data folder> --test_type train_distribution_mean_blur_torch_gmm_ablation
python AFHQ_128_test.py --gmm_cluster 1 --noise 0.002 --discrete --time_steps 300 --blur_size 27 --blur_std 0.01 --blur_routine Exponential --sampling_routine x0_step_down --save_folder <Path to save results> --load_path <Path to load models> --data_path <Path to data folder> --test_type train_distribution_mean_blur_torch_gmm_ablation
```
### Animorphosis

The code for these models is in the `demixing-diffusion-pytorch` folder.

#### Unconditional Generation

*Training*
```
python AFHQ_128_to_celebA_128.py --time_steps 200 --sampling_routine x0_step_down --save_folder <path to save models> --data_path_start <Path to starting data manifold> --data_path_end <Path to ending data manifold>
```
*Testing*
```
python AFHQ_128_to_celebA_128_test.py --time_steps 200 --sampling_routine x0_step_down --save_folder <Path to save images> --load_path <Path to load model> --data_path_start <Path to starting data manifold> --data_path_end <Path to ending data manifold> --test_type test_sample_and_save_for_fid
```

### Inpainting

The code for the conditional generation is in the `defading-diffusion-pytorch` folder, and the code for unconditional generation is in the `defading-generation-diffusion-pytorch` folder. For Gaussian masking, the arguments that specify the degradation schedule are `--time_steps`, `--kernel_std`, and `fade_routine`. See the help strings in the training scripts for details on what these arguments specify.

#### Conditional Generation

*Training*
```
python mnist_train.py --time_steps 50 --save_folder <path to save models> --discrete --sampling_routine x0_step_down --train_steps 700000 --kernel_std 0.1 --fade_routine Random_Incremental --data_path <Path to data folder>
python cifar10_train.py --time_steps 50 --save_folder <path to save models> --discrete --sampling_routine x0_step_down --train_steps 700000 --kernel_std 0.1 --fade_routine Random_Incremental --data_path <Path to data folder>
python celebA_train.py --time_steps 100 --fade_routine Incremental --save_folder <path to save models> --sampling_routine x0_step_down --train_steps 350000 --kernel_std 0.2 --image_size 128 --dataset celebA --data_path <Path to data folder>
```
*Testing*
```
python mnist_test.py --time_steps 50 --save_folder test_mnist --discrete --sampling_routine x0_step_down --kernel_std 0.1 --initial_mask 1 --image_size 28 --fade_routine Random_Incremental --load_path <Path to load model> --data_path <Path to data folder> --test_type test_data 
python cifar10_test.py --time_steps 50 --save_folder test_cifar10 --discrete --sampling_routine x0_step_down --kernel_std 0.1 --initial_mask 1 --image_size 32 --fade_routine Random_Incremental --load_path <Path to load model> --data_path <Path to data folder> --test_type test_data
python celebA_test.py --time_steps 100 --fade_routine Incremental --save_folder test_celebA --sampling_routine x0_step_down --kernel_std 0.2 --initial_mask 1 --image_size 128 --dataset celebA --load_path <Path to load model> --data_path <Path to data folder> --test_type test_data
```

#### Unconditional Generation

*Training*
```
python celebA_128.py --reverse --kernel_std 0.05 --initial_mask 1 --time_steps 750 --sampling_routine x0_step_down --save_folder <Path to save models> --data_path <Path to data folder>
```

*Testing*
```
python celebA_constant_128_test.py --reverse --kernel_std 0.05 --time_steps 750 --sampling_routine x0_step_down --save_folder <Path to save images> --data_path <Path to data folder> --load_path <Path to load model> --test_type test_sample_and_save_for_fid
```

### Super-Resolution

The code for these models is in the `resolution-diffusion-pytorch` folder. For downsampling, the arguments that specify the degradation schedule are `--time_steps`, and `resolution_routine`. See the help strings in the training scripts for details on what these arguments specify.

#### Conditional Generation

*Training*
```
python mnist_train.py --time_steps 3 --resolution_routine 'Incremental_factor_2' --save_folder <Path to save models>
python cifar10_train.py --time_steps 3 --resolution_routine 'Incremental_factor_2' --save_folder <Path to save models>
python celebA_128.py --time_steps 4 --resolution_routine 'Incremental_factor_2' --save_folder <Path to save models>
```
*Testing*
```
python mnist_test.py --time_steps 3 --train_routine 'Final' --sampling_routine 'x0_step_down' --resolution_routine 'Incremental_factor_2' --save_folder <Path to save images> --load_path <Path to load model> --test_type test_data
python cifar10_test.py --time_steps 3 --train_routine 'Final' --sampling_routine 'x0_step_down' --resolution_routine 'Incremental_factor_2' --save_folder <Path to save images> --load_path <Path to load model> --test_type test_data
python celebA_test.py --time_steps 4 --train_routine 'Final' --sampling_routine 'x0_step_down' --resolution_routine 'Incremental_factor_2' --save_folder <Path to save images> --load_path <Path to load model> --test_type test_data
```

#### Unconditional Generation

*Training*
```
python celebA_test.py --time_steps 4 --train_routine 'Final' --sampling_routine 'x0_step_down' --resolution_routine 'Incremental_factor_2' --save_folder <Path to save images> --load_path <Path to load model> --test_type test_data
```
*Testing*
```
python celebA_test.py --time_steps 4 --train_routine 'Final' --sampling_routine 'x0_step_down' --resolution_routine 'Incremental_factor_2' --save_folder <Path to save images> --load_path <Path to load model> --test_type test_data
```

### Snowification

The code for these models is in the `snowification` folder. This code stems from the existing <a href="https://pypi.org/project/imagenet-c/">ImageNet C</a> repository. The arguments that specify the degradation schedule are `--time_steps`, `--snow_level`, `--random_snow`, and `--fix_brightness`. See the help strings in the training scripts for details on what these arguments specify.

#### Conditional Generation

*Training*
```
python train.py --dataset cifar10 --time_steps 200 --forward_process_type ‘Snow’ --snow_level 3 --exp_name <exp_name>  --dataset_folder <path-to-dataset> --random_snow --fix_brightness  --sampling_routine x0_step_down
python train.py --dataset celebA --time_steps 200 --forward_process_type ‘Snow’ --snow_level 4 --exp_name <exp_name> --dataset_folder <path-to-dataset> --random_snow --fix_brightness  --sampling_routine x0_step_down
```
*Testing*
```
python test.py --dataset cifar10 --time_steps 200 --forward_process_type ‘Snow’ --snow_level 3 --exp_name <exp_name> --dataset_folder <path-to-dataset> --random_snow --fix_brightness --resume_training --sampling_routine x0_step_down --test_type test_data --order_seed 1
python test.py --dataset celebA --time_steps 200 --forward_process_type ‘Snow’ --snow_level 4 --exp_name <exp_name> --dataset_folder <path-to-dataset> --random_snow --fix_brightness --resume_training --sampling_routine x0_step_down --test_type test_data --order_seed 1
```

### Colorization

The code for these models is in the `decolor-diffusion` folder. For color desaturation, the arguments that specify the degradation schedule are `--time_steps`, `decolor_total_remove`, and `decolor_routine`. See the help strings in the training scripts for details on what these arguments specify.

#### Conditional Generation

*Training*
```
python train.py --dataset cifar10 --time_steps 20 --forward_process_type ‘Decolorization’ --exp_name <exp_name> --decolor_total_remove --decolor_routine ‘Linear’ --dataset_folder <path-to-dataset>
python train.py --dataset celebA --time_steps 20 --forward_process_type ‘Decolorization’ --exp_name <exp_name> --decolor_total_remove --decolor_routine ‘Linear’ --dataset_folder <path-to-dataset>
```
*Testing*
```
python test.py --dataset cifar10 --time_steps 20 --forward_process_type ‘Decolorization’ --exp_name <exp-name>  --decolor_total_remove --decolor_routine ‘Linear’ --dataset_folder <path-to-dataset> --sampling_routine x0_step_down --test_type test_data --order_seed 1
python test.py --dataset celebA --time_steps 20 --forward_process_type ‘Decolorization’ --exp_name <exp-name>  --decolor_total_remove --decolor_routine ‘Linear’ --dataset_folder <path-to-dataset> --sampling_routine x0_step_down --test_type test_data --order_seed 1
```