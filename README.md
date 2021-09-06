# Denoising Hyperspectral Images by Unsupervised Deep Learning

Implementation of the work "Denoising Hyperspectral Images by Unsupervised Deep Learning".


### Requirements
- python = 3.6
- pytorch = 0.4
- numpy
- scipy
- matplotlib
- scikit-image

### Prepare the data
- The hyperspectral data should be contained in `*.mat` files. For the simulated data set, the noisy data and ground truth can be separately stored in a `*.mat` file with variable name `y` and `z`, respectively.
- Specify a path to the file and name of the variable to read.</br>
   For example, if noisy and clean data are contained in `denoising_DC_Mall_case3.mat` with variable name `y` and `z`:
   ```
   file_name  = 'data/denoising/denoising_DC_Mall_case3.mat'
   mat = scipy.io.loadmat(file_name)
   img_np = mat["y"]
   img_noisy_np = mat["z"]
   ```
- Use custom code or one of the `*.m` files located at `data/%task%/` to generate `*.mat` file.

### Run the code

- ./run.sh

<br>

### Acknowledgement
The implementation of this work is based on [original Deep Image Prior code by Dmitry Ulyanov](https://github.com/DmitryUlyanov/deep-image-prior) and [O Sidorov, JY Hardeberg. Deep Hyperspectral Prior: Denoising, Inpainting, Super-Resolution](https://arxiv.org/abs/1902.00301).

-------
#### Please, kindly cite the paper if you use the code!
