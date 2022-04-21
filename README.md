# Matlab Image Denoiser

The entire work is based on the article "Extracting and Composing Robust Features with Denoising Autoencoders", which proposes the creation of a neural network with the ability to reconstruct distorted images, following different specifications.
The goal is to create the structure described in the article, in order to denoise an image dataset, to which noise (of different types) has been appropriately applied.
The MNIST dataset was used, so as to have a considerable number of redundant images, in order to be able to carry out a good training phase and, therefore, obtain better results in the reconstruction phase.
Following the article, two fully connected feed-forward neural networks have been implemented which are respectively the autoencoder and the denoiser.

## Dependencies
The only thing missing here is the MNIST Dataset. The problem here is that while doing the commit by command line github doesn't throw a tantrum but
when uploading files directly from the browser (as this repo has been created) it limits the file dimension to 25MB.

So you just need to obtain the dataset from [MNIST](http://yann.lecun.com/exdb/mnist/).
Once you downloaded the files, put them into a folder called "dataset_mnist" in order to make the code work with it.

## Usage
In order to start the denoising process, it is only required to execute the main.


## Contributing
Marco Urbano, [marcourbano.me](https://marcourbano.me).  
Luigi Urbano, [0xUrbz](https://github.com/0xUrbz).  
Ciro Brandi.  [Ciro Brandi](https://github.com/Brandi-Ciro)

## License
[MIT](https://choosealicense.com/licenses/mit/)
-
