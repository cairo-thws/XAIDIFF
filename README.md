# GitHub Repository for XAIDIFF
A framework for local example-based explanations with generative models.
The code is published at [Github](https://github.com/cairo-thws/XAIDIFF)
The pretrained models can be found at [Huggingface](https://huggingface.co/cairo-thws/XAIDIFF).

![XAIDIFF Framework](XAIDIFF.png)



# About this repository
This repository contains the code for the XAIDIFF framework.
The framework is designed to provide local example-based explanations for image- and object classifiers with the help of generative models.
Using a diffusion model, the framework generates counterfactual examples that are similar to the input image but are classified differently by the classifier.
The framework is designed to be modular and can be used with different generative models and classifiers.

# Installation and usage

## Conda environment
First, create a new conda environment and activate it.
```
conda env create -f environment.yml
conda activate xaidiff
```

## Sampling
Per default the project allows you to sample from any pretrained diffusion model with any pretrained classifier.
The `SportBallsDataset` is a simple example dataset that can be used to test the framework.
If you want to add your own classifier and dataset, just inspect the `SportBallsDataset`.

The file `experiment_sampling.py` contains the core functionality of this project and can be used to sample from the models.
There are many configurable parameters at the head of the file that can be used to adjust the sampling process.
It will generate counterfactual examples for the given input image and classifier.

The results contain the following information:
- An image with four rows: the original image, the counterfactual sample, an affirmative sample and an adversarial sample.
- Different metrics for the generated samples, including LP distances and bits per dimension. This allows to compare the generated samples with the original image.

The file `experiment_sampling_sportsballs.py` contains an example of how to adapt the original sampling process to the `SportBallsDataset`.
To execute it, download the pretrained sportballs classifier and diffusion model.

