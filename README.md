# Dog classifier
This project uses transfer learning to build a CNN pre-trained on ImageNet, with ResNet-18 architecture, to classify dog breeds. The training dataset is the [Stanford Dogs Dataset](http://vision.stanford.edu/aditya86/ImageNetDogs/). The [fastai](https://github.com/fastai/fastai) library (v2), which is a high-level library on top of PyTorch, was used for creating the model.

A 10% test set (holdout set) was set aside and hidden from fastai. The validation set was 20% of the remaining data. Both the test set and validation set were chosen as stratified samples to maintain class balance in the datasets.

Zero-padding was used when resizing images to a common size in order to maintain the aspect ratio of the original image. The default fastai augmentation transforms (; see ) were used during training to improve the generalization performance of the model. Note that this augmentation strategy does not increase the number of images in each epoch; it just randomly alters each image in each batch, so that a given image will be viewed differently in each epoch.

A learning rate was chosen using the `lr_find` feature in fastai, which starts with a very small learning rate, uses that for one mini-batch, calculates the losses, then increases the learning rate and repeats until the loss gets worse. Then the learning rate selected is the part of the loss vs. learning rate curve where the loss decreases the quickest before increasing again. See the fastai book, chapter 5 (search for "learning rate finder" [here](https://github.com/fastai/fastbook/blob/master/05_pet_breeds.ipynb)).

The holdout set was evaluated on the final model, given a test set accuracy of 53.76%, compared to a random chance of 1/120 = 0.83% (because there are 120 dog breeds).

Finally, I used the trained model to make predictions of breeds on dogs that I know, as well as the top 5 after the best guess.

Future work:
* run model with explicit normalization of images (should be already baked-in, but worth testing)
* add images for dog breeds not in the stanford dogs dataset
* add a nearest neighbors analysis to find the nearest dog images in the dataset, based on the CNN's last-layer embedding (before the FC layer)
* an object detection model using the bounding boxes given in the Stanford dogs dataset

![example of classification](example.png)
