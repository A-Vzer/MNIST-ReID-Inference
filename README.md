# MNIST reidentification via 2D embedding space and inference on own handwriting 

This code will:
1. Use two dataloaders; one for mini batch training and the other to track 20 samples of each digit
2. Train the model (on gpu and with a set seed) and plot the 2 dimensional embedding space every epoch
3. Print accuracy and top-20 Reidentification accuracy. Wherre the reidentification accuracy is found with the nearest neighbour algorithm
4. Plot loss per batch, accuracy per batch, accuracy per epoch and ReID accuracy per epoch
5. Finally, it will run inference on the handwritten digits which can be found in the 'model/digits/' folder and show the top 5 selection of every digit. The digits in the folder will be preprocessed with openCV before passing it through the model.

For any questoins feel free to contact me on discord: Vzer#7201
