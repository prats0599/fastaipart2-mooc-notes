# fastaipart2-mooc-notes
This repository contains my notes & code for the [(Part2)Deep Learning from the foundations](https://course.fast.ai/part2) course by fastai.    
  
The description of each file is as below:
- [exp](https://github.com/prats0599/fastaipart2-mooc-notes/tree/master/exp): contains all relevant code that we wrote in the notebooks at one place so that it's easier to import them and use as a library.
- [00_exports](00_exports.ipynb): converting notebooks to script so as to integrate them all easily.
### Computer Vision
- [01_matmul](01_matmul.ipynb): Matrix multiplications and broadcasting.
- [02_fully_connected](02_fully_connected.ipynb): Coded the forward and backward passes in a neural network. Studied and implemented Xavier and kaiming initialization. Also, recreated the nn.Module and nn.Linear class from the Pytorch library.
- [03_minibatch_training](03_minibatch_training.ipynb): Creating an optimizer, Dataset and Dataloader(fastai) class; Create a basic training loop to train the model.
- [04_callbacks](04_callbacks.ipynb): Adding callbacks to the training loop to make it customizable as per requirement.
- [05_anneal](05_anneal.ipynb) : Implemented a learning rate annealer([One cycle policy by Leslie smith](https://arxiv.org/pdf/1803.09820.pdf)) using the callback system created and achieved better accuracy.
- [05a_foundations](05a_foundations.ipynb): ways to use callbacks, special methods in Python(basics), when not to use softmax 
- [05b_early_stopping](05b_early_stopping.ipynb): Implemented Early Stopping during training on the model using a callback.
- [06_cuda_cnn_hooks](06_cuda_cnn_hooks.ipynb): Training on the GPU; Pytorch hooks and tips on initializing weights of your model.
- [07_batchnorm](07_mybatchnorm.ipynb): Studied and Implemented Batchnorm, Instance Norm, Layer Norm and Group Norm.
- [07a_lsuv](07a_mylsuv.ipynb): Implemented the [All you need is a good init](https://arxiv.org/abs/1511.06422) algorithm(Layerwise Sequential Unit variance).
- [08_datablock](08_mydatablock.ipynb):Recreating parts of the DataBlock api present in the fastai library.
- [09_ optimizers](09_myoptimizers.ipynb): Implemented optimizers in a flexible way. Started with implementing plain SGD, followed by adding momentum and weight decay to it. We then implemented the ADAM optimizer from scratch and finally ended with the LAMB optimizer which was mentioned [here](https://arxiv.org/abs/1904.00962).
- [09b_learner](09b_mylearner.ipynb): Refactored a bit of code and incorporated the learner and Runner class previously created into one Learner class.
- [09c_addprogressbar](09c_addprogressbar.ipynb): Added the progress bar graphic(similar to tqdm) which is depicted during training.
- [10_augmentation](10_myaugmentation.ipynb): Studied and implemented several data augmentation techniques(for images) present in the fastai library such as zooming, flipping, RandomResizeCrop and Perspective warping. Applied Data augmentation to images on the GPU which improved execution time by magnitudes.
- [10b_mixup_data_augmentation](10b_mixup_augmentation.ipynb): Studied & Implemented MixUp Data augmentation and Label Smoothing which improved model accuracy on training.
- [10c_FP16](10c_FP16.ipynb): Applied Mixed Precision Training to the model using [Nvidia's Apex libray](https://github.com/NVIDIA/apex)
- [11_train_imagenette](11_train_imagenettee.ipynb): Studied and implemented different types of resnets from the [Bag of tricks for Image classfication paper](https://arxiv.org/abs/1812.01187)
- [11a_transfer_learning](11a_transfer_learning.ipynb): Implemented transfer learning from scratch and used a model pretrained on the Imagewoof dataset(10 classes of dog breeds) and trained it on the [Cats and Dogs breed classifciation Oxford dataset](https://www.kaggle.com/zippyz/cats-and-dogs-breeds-classification-oxford-dataset).  
       
### NLP
- [12_text](12_text.ipynb):
- [12a_awd_lstm](12a_awd_lstm.ipynb):
- [12b_lang_model_pretrained](12b_lang_model_pretrained.ipynb):
- [12c_ulmfit_from_scratch](12c_ulmfit_from_scratch):

  
## Research Papers Implemented:
* [Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification](https://arxiv.org/abs/1502.01852)
* [Understanding the difficulty of training deep feedforward neural networks](http://proceedings.mlr.press/v9/glorot10a.html)
* [Fixup Initialization: Residual Learning Without Normalization](https://arxiv.org/abs/1901.09321)
* [All you need is a good init](https://arxiv.org/abs/1511.06422)
* [Exact solutions to the nonlinear dynamics of learning in deep linear neural networks](https://arxiv.org/abs/1312.6120)
* [Self-Normalizing Neural Networks](https://arxiv.org/abs/1706.02515)
* [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167)
* [Layer Normalization](https://arxiv.org/abs/1607.06450)
* [Instance Normalization: The Missing Ingredient for Fast Stylization](https://arxiv.org/abs/1607.08022)
* [Group Normalization](https://arxiv.org/abs/1803.08494)
* [Revisiting Small Batch Training for Deep Neural Networks](https://arxiv.org/abs/1804.07612)
* [All you need is a good init](https://arxiv.org/abs/1511.06422)
* [Decoupled Weight Regularization](https://arxiv.org/abs/1711.05101.pdf)
* [L2 Regularization versus Batch and Weight Normalization](https://arxiv.org/abs/1706.05350)
* [Norm matters: efficient and accurate normalization schemes in deep networks](https://arxiv.org/abs/1803.01814)
* [Three Mechanisms of Weight Decay Regularization](https://arxiv.org/abs/1810.12281)
* [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980)
* [Reducing BERT Pre-Training Time from 3 Days to 76 Minutes (LAMB optimizer paper)](https://arxiv.org/abs/1904.00962)
* [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842)
* [mixup: Beyond Empirical Risk Minimization](https://arxiv.org/abs/1710.09412)
* [Rethinking the Inception Architecture for Computer Vision (label smoothing is in part 7)](https://arxiv.org/abs/1512.00567)
* [Bag of Tricks for Image Classification with Convolutional Neural Networks (XResNets)](https://arxiv.org/abs/1812.01187)
* [Regularizing and Optimizing LSTM Language Models (AWD-LSTM)](https://arxiv.org/abs/1708.02182)
* [Universal Language Model Fine-tuning for Text Classification (ULMFiT)](https://arxiv.org/abs/1801.06146) 
  
  
All credits to Jeremy Howard and the [fastai](https://www.fast.ai/) team!
