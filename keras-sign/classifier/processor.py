import os, sys

def save_keras(model_name, kmodel, ep):
    modelJSON = kmodel.to_json()
    with open(os.path.join("classifier","{0}-{1}.json".format(model_name, ep), "w")) as jfile:
        jfile.write(modelJSON)

    kmodel.save_weights(os.path.join("classifier", "{0}-{1}.h5".format(model_name, ep)))
    print("Saved model to %s" % os.path.join("classifier", "{0}-{1}".format(model_name, ep)))

def load_keras(ep):
    jfile = open(os.path.join(model_path, "{0}-{1}.json".format(model_name, ep)))
    modelJSON = jfile.read()
            
    kmodel = model_from_json(modelJSON)
    kmodel.load_weights(os.path.join(model_path, "{0}-{1}.h5".format(model_name, ep)))
    print("Loaded Keras model from {0}!".format(os.path.join(model_path)))
    return kmodel