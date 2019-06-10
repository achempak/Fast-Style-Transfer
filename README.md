Description
===========
This project employs deep neural networks to transfer the style of one image to another. The repository contains two subdirectories that each implement an approach to execute the style transfer process. 

Slow_Style implements the solution to an unconstrained optimization problem posed by Gatys et. al. in "A Neural Algorithm for Style Transfer": the problem models the process as a weighted linear expression in the feature and style representation losses corresponding to a content and a style image respectively. Fast_Style implements the machine learning approach to solving the same problem, devised by Justin Johnson et. al., with considerable changes to the architecture used by Gatys, but offering a significant speedup to computation.

This work is a culmination of the efforts of Aditya Chempakasseril, Aisha Dantuluri and Soumyaraj Bose, together named as The Outliers.

Requirements
============
The setup in the repository is self-sufficient; no files or modules need to be separately installed. There is a demo for real-time style transfer in the Fast_Style directory.

Code Organization
=================
*Slow_Style:*

slow_model.py     --  Contains the VGG-19 neural network architecture and functionality to solve the problem posed by Gatys

slow_utils.py     --  Contains utility functions to preprocess and display images and plots for the losses

Slow_Style.ipynb  --  Runs the computing framework for slow style transfer and generates results seen in Sections 3 and 4. This file also functions as the necessary demo file for this model.

*Fast_Style:*

transformer.py    --  Contains the neural network to transform the content image and prior to feeding to the loss network

lossnet.py        --  Contains the pretrained VGG-16 network to extract the feature reconstruction and style losses from a content and a style image respectively

utils.py          --  Contains the utility functions to preprocess and display images and plots for the the losses, as well as load and save the trained models (of the image transformation network) at checkpoints

Realtime.ipynb    --  Runs the computing framework for fast style transfer and generates the results seen in Sections 3 and 4

Fast_Style_Demo.ipynb -- Runs a demo with pretrained networks
