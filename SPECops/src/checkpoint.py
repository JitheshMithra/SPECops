import pickle

#bundles trained weights together with the run config so whoever loads it later knows exactly which architecture the weights belong to
def saveCheckpoint(path, params, config):
    with open(path, "wb") as f:
        pickle.dump({"params": params, "config": config}, f)

def loadCheckpoint(path):
    with open(path, "rb") as f:
        blob = pickle.load(f)
    return blob["params"], blob["config"]
