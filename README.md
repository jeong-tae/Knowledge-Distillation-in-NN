# Knowledge-Distillation-in-NN

Knowledge-Distillation(KD) is a simple way to compress model while keeping the performance of original model. This repository provide a implementation of the paper ["Distilling the Knowledge in a Neural Network"](https://arxiv.org/pdf/1503.02531.pdf) with some changes. Please check the references for detailed explanations.


## Requirements
  - python 3.6 >
  - pytorch 1.3 >

## Structure
    .
    |--experiments (it will include the scripts to run, results of train and test)
    |--train_net.py (main solver to train the model)
    |--data/
    |    |--data_loader.py (Data queue module)
    |--models/
    |    |--Loss.py (Loss functions that are used in this project)
    |--engine/
    |    |--trainer.py
    |    |--inference.py
    |    |--solver.py
    |--utils/
    |    |--logger.py (module that can visualization of image and training plot)  
    |    |--checkpointer.py
    |    |--measure.py

You should make clear that all directory and files are located correctly
  
## TODO
  - [ ] Check all requirements to run
  - [ ] Additional feature to improve baseline(Hinton's 15)
  - [x] Train teachers

## Usage

I added two scripts. One for training the teacher model and another for trainingthe student using teacher.  
Specify your teacher model in models and build_model if you have own your model and put arguments correctly to run.

### How to train?
```bash
$ ./experiments/exp1/train.sh
```

### How to test?


## Performance
Table will be appear

## To track your training progress
```bash
$ tensorboard --logdir=experiment/ --port=6666
```

and go to 'localhost:6666' on webbrowser. You can see the accuracy and loss graph

## References
  - ["Distilling the Knowledge in a Neural Network"](https://arxiv.org/pdf/1503.02531.pdf)
  - ["Python implementaion of hinton's KD"](https://github.com/peterliht/knowledge-distillation-pytorch)
    - I refered CS230 report to understand loss function(Cross entropy + KL_div)
