Description
===========
This project employs deep neural networks to transfer the style of one image to another. The repository contains two subdirectories that each implement an approach to execute the style transfer process. 

Slow_Style implements the solution to an unconstrained optimization problem posed by Gatys et. al. in "A Neural Algorithm for Style Transfer": the problem models the process as a weighted linear expression in the feature and style representation losses corresponding to a content and a style image respectively. Fast_Style implements the machine learning approach to solving the same problem, devised by Justin Johnson et. al., with considerable changes to the architecture used by Gatys, but offering a significant speedup to computation.

This work is a culmination of the efforts of Aditya Chempakasseril, Aisha Dantuluri and Soumyaraj Bose, together named as The Outliers.

Requirements
============
The setup in the repository is self-sufficient; no files or modules need to be separately installed.

Code Organization
=================
Slow_Style:

slow_model.py     --  Contains the VGG-19 neural network architecture and functionality to solve the problem posed by Gatys <\br>
slow_utils.py     --  Contains utility functions to preprocess and display images and plots for the losses <\br> 
Slow_Style.ipynb  --  Runs the computing framework for slow style transfer and generates the results seen in Sections 3 and 4 <\br>

Fast_Style:

transformer.py    --  Contains the neural network to transform the content image and prior to feeding to the loss network \n
lossnet.py        --  Contains the pretrained VGG-16 network to extract the feature reconstruction and style losses from a content and a style image respectively \n
utils.py          --  Contains the utility functions to preprocess and display images and plots for the the losses, as well as load and save the trained models (of the image transformation network) at checkpoints \n
Realtime.ipynb    --  Runs the computing framework for fast style transfer and generates the results seen in Sections 3 and 4 \n
