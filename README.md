# fastaipart2-mooc-notes
This repository contains my notes and code that were taught in [Part2:Deep Learning from the foundations](https://course.fast.ai/part2) course.    
  
The description of each file is as below:
- [exp](https://github.com/prats0599/fastaipart2-mooc-notes/tree/master/exp): contains all relevant code that we wrote in the notebooks at one place so that it's easier to import them and use as a library.
- [00_exports](00_exports.ipynb): converting notebooks to script so as to integrate them all.
- [01_matmul](01_matmul.ipynb): Matrix multiplications and broadcasting
- [02_fully_connected](02_fully_connected.ipynb): Creating the forward and backward passes in the neural network. Also, recreate the nn.Module and nn.Linear class from the Pytorch library.
- [03_minibatch_training](03_minibatch_training.ipynb): Creating an optimizer, Dataset and Dataloader(fastai) class; Create a basic training loop and train the model.
- [04_callbacks](04_callbacks.ipynb): Adding callbacks to the training loop to make it customizable as per requirement.
- [05_anneal](05_anneal.ipynb) : Create and implement a learning rate annealer using the callback system created.
- [05a_foundations](05a_foundations.ipynb): ways to use callbacks, special methods in Python, when not to use softmax 
- [05b_early_stopping](05b_early_stopping.ipynb): implementing Early Stopping during training on our model using a callback.
- [06_cuda_cnn_hooks](06_cuda_cnn_hooks.ipynb): Training on the GPU; Pytorch hooks and tips on initializing weights of your model.
- [07_batchnorm](07_mybatchnorm.ipynb): Batchnorm, Instance Norm, Layer Norm, Group Norm
- [07a_lsuv](07a_mylsuv.ipynb): Implementing the [All you need is a good init](https://arxiv.org/abs/1511.06422) algorithm(Layerwise Sequential Unit variance).
- [08_datablock](08_mydatablock.ipynb):Recreating parts of the DataBlock api present in the fastai library.
   
     
All credits to Jeremy Howard and the [fastai](https://www.fast.ai/) team!
