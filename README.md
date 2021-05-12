# Stop_start_training
Often times we need to interrupt the training process of our deep neural network in between epochs before completion for multiple reason,for instance Power failure ,for tuning 
the learning rate in case loss/accuracy of our network plateaued. Therefore it is essential for us to save our model at a regular intervals so that we dont have to retrain our 
network from beginning.Keras callbacks API provides us with classes which can be extended and manupulated.Methods within classes are called at the beginning and end of every epoch 
,batches within an epoch and even at the beginning and end of the training process itself. We can manupulate these methods at our convinience to accomplice specific task. Here i 
have leveraged these methods to save the model every 5 epochs by extending `tensorflow.keras.callbacks.Callback` baseclass and to save the Loss/Accuracy and Logs every single epoch. 
These serialized models can be loaded from the disc everytime we have to restart our training process from last finished point.
