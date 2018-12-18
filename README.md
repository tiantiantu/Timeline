# Timeline model 
This repository contains codes for Timeline model in paper:
* **Bai, T., Zhang, S., Egleston, B.L., Vucetic, S., Interpretable Representation Learning for Healthcare via Capturing Disease Progression through Time, KDD, 43-51, 2018.**

I used the following environment for the implementation:
* python==3.7.0
* torch==0.4.1
* numpy==1.15.1
* sklearn==0.19.2

To run the model, three files are required: 
* visitfile: a nested list including patients which are lists including visits which are list including codes.
* labelfile: a list including labels.
* gapfile:  a nested list including patients which are lists including time interval between a past visit and the visit of prediction.

As an example, assume dataset contains two patients A and B: 

Patient A has three visits: visit 1 contains two codes: 174, 250; visit 2 contains one code: 274; visit 3 is associated with a label 1. The time interval between visit 1 and visit 3 is 100; the time interval between visit 2 and visit 3 is 15.

Patient B has two visits: visit 1 contains one code 350; visit 2 is associated with a label 2. The time interval between visit 1 and visit 2 is 3.

Then visitfile is a npy file of a list [  [ ['174', '250'], ['274'] ]  ,  [ ['350'] ]  ]

labelfile is a npy file of a list [1,2]

gapfile is a npy file of a list [ [100, 15], [3] ]

The following example command will run the code:

``python Timeline.py visitfile.npy labelfile.npy gapfile.npy --EMBEDDING_DIM=80 --HIDDEN_DIM=80 --ATTENTION_DIM=80 --EPOCH=100 --batchsize=48 --dropoutrate=0.2``
