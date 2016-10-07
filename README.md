# tf-devanagri
Convolutional Neural Network to identify Devanagri characters implemented in Tensorflow.

### Data

* n_classes = 4
* Size = 320x320
* Format = Binary, PNG


### Results and Preprocessing

* 94.6 +/- 0.07 % accuracy on test data.
* 30 epochs in 30 mins of training time on a 4 GB NVIDIA GT920M GPU and i7 CPU.
* 10x downsampling to get 32x32 images.
* 3-iterations of binary dilation to make characters thicker.

### Network

#### Architecture

* Conv2d 7x7 1
* Conv2d 5x5 32
* Conv2d 5x5 64
* Conv2d 3x3 128
* Conv2d 3x3 256
* Dense  256x2x2
* Dense  1024
* Softmax 104 = `n_classes`

#### HyperParameters

* Learning Rate = 0.003
* Momentum = 0.9
