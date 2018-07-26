import os, sys

def save_keras(model_name, kmodel, ep):
    modelJSON = kmodel.to_json()
    with open(os.path.join("classifier","{0}-{1}.json".format(model_name, ep), "w") as jfile:
        jfile.write(modelJSON)

    kmodel.save_weights(os.path.join("classifier", "{0}-{1}.h5".format(model_name, ep)))
    if self.verbose: print("Saved model to %s" % os.path.join("classifier", "{0}-{1}".format(model_name, ep)))
    else:
        raise Exception("Set model path.")

def load_keras(self, ep):
    jfile = open(os.path.join(self._model_path,"{0}-{1}".format(self._model_name, ep), "{0}-{1}.json".format(self._model_name, ep)))
    modelJSON = jfile.read()
            
    kmodel = model_from_json(modelJSON)
    kmodel.load_weights(os.path.join(self._model_path,"{0}-{1}".format(self._model_name, ep), "{0}-{1}.h5".format(self._model_name, ep)))
    if self.verbose: print("Loaded Keras model from {0}!".format(os.path.join(self._model_path,"{0}-{1}".format(self._model_name, ep))))
        return kmodel
    else:
        raise Exception("Set model path.")