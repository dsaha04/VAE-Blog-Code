# VAE-Blog-Code

This is the code associated with the blog post linked [here](https://dsaha04.github.io/blog/2024/VAEs&EM/). It implements a simple reconstruction task on the MNIST dataset using exactly the theoretical training objective listed in our paper.

It can be extended to doing generation, using learned activations to restrict the output space, or potentially using a deterministic decoder while in evaluation mode.

## How to use?

It is pretty simple. Simply create the following conda environment and install the specified dependencies.

```
conda create -n vae python=3.10
conda activate vae
pip install torch torchvision matplotlib
```
Simply run
```
python vae.py
```
to start training and to see some reconstructions!

## Citation
The code was written by me and the paper by myself and Sunay Joshi. Below we provide a citation to the paper for formality sake.

```
@article{sahajoshivaeandem2024,
  title   = "Variational Autoencoders and the EM Algorithm",
  author  = "Saha, Dwaipayan", "Joshi, Sunay"
  year    = "2024",
  month   = "August",
  url     = "https://dsaha04.github.io/blog/2024/VAEs&EM/"
}
```