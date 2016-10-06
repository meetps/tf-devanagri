# tf-devanagri
Convolutional Neural Network to identify Devanagri characters implemented in Tensorflow.

### Data

* n_classes = 4
* Size = 320x320
* Format = Binary, PNG


### Results and Preprocessing

* 92.4 +/- 0.07 % accuracy on test data.
* 50 epochs in 2 hrs of training time on a 4 GB NVIDIA GT920M CPU and i7 CPU.
* 4x downsampling to get 80x80 images.
* 3-iterations of binary dilation to make characters thicker.

### Network

#### Architecture 

* Conv2d 7x7 1
* Conv2d 5x5 32
* Conv2d 5x5 64
* Conv2d 3x3 128
* Conv2d 3x3 256
* Dense  256x5x5
* Dense  2048
* Softmax 104 = `n_classes`

#### HyperParamters

* Learning Rate = 0.001
* Momentum = 0.9
