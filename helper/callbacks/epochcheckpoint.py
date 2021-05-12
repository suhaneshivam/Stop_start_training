# import packages
from tensorflow.keras.callbacks import Callback
import os

#This would serialized the model at every 5 epoch.If the model is already save it will overwrite and create a new model.
class EpochCheckpoint(Callback):

    def __init__(self, outputPath, every = 5, startAt = 0):
        # call the parent instructor
        super(EpochCheckpoint, self).__init__()

        # store the base output path for the model, the number of
        # epochs that must pass before the model is serialized to
        # disk and the current epoch value
        self.outputPath = outputPath
        self.every = every
        self.intEpoch = startAt

    def on_epoch_end(self, epoch, logs = None):
        # check to see if the model should be serialized to disk
        if (self.intEpoch + 1) % self.every == 0:
            p = os.path.sep.join([self.outputPath,
                "epochs_{}.hdf5".format(self.intEpoch + 1)])
            self.model.save(p, overwrite = True)

        # increment internal epoch counter
        self.intEpoch += 1
