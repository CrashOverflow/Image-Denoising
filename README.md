# Matlab Image Denoiser (provided with MNIST Dataset)

The entire work is based on the article "Extracting and Composing Robust Features with Denoising Autoencoders", which proposes the creation of a neural network with the ability to reconstruct distorted images, following different specifications.
The goal is to create the structure described in the article, in order to denoise an image dataset, to which noise (of different types) has been appropriately applied.
The MNIST dataset was used, so as to have a considerable number of redundant images, in order to be able to carry out a good training phase and, therefore, obtain better results in the reconstruction phase.
Following the article, two fully connected feed-forward neural networks have been implemented which are respectively the autoencoder and the denoiser.

## Usage
In order to start the denoising process, it is only required to execute the main.

## Contributing
Marco Urbano, [marcourbano.me](https://marcourbano.me).  
Luigi Urbano, [0xUrbz](https://github.com/0xUrbz).  
Ciro Brandi.  

## License
[MIT](https://choosealicense.com/licenses/mit/)
