\- N. Abinash, Partheban KV

### Dataset: 
- Taken from kaggle (https://www.kaggle.com/datasets/enalis/tomatoes-dataset)
- Contains 7226 images of tomatoes in four different states: Unripe, Ripe, Old, and Damaged.
- For this model only images in the dataset within the subdirectories "Ripe" and "Unripe" are used for training the model.
- 90/5/5 Split (total: 3949)
	Ripe: 2195
	  Train: 1975
	  Test: 110
	  Validation: 110
	  
	Unripe: 1754
	 Train: 1585
	 Test: 84
	 Validation: 85

### Model:
-  The model is VGG16 connected with a Multilayer Perceptron block which consists of a fully connected layer and an output layer.
- Images of size 224 x 224 with 3 channels are fed into the frozen convolutional layers of VGG16. This is then fed into the last two unfrozen convolutional layers for fine tuning. This is connected to a MLP block.
- The output layer has two nodes (for ripe and unripe) with softmax activation function.
- Hyperparameters:
	- Drop Out: 0.3
	- Total Epochs:100
	- Early Stopping: 20
	- Optimizer: Adam
	- BatchSize: 16
	- Initial Learning Rate: 1e-3
- This model is based on this existing paper: https://www.ijisae.org/index.php/IJISAE/article/view/2538/1121


### Next Step/ Suggestions:
- Variation in the model to make it a novel approach.
