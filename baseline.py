import yaml
import json

import pickle
import pandas as pd
import tensorflow as tf
from pathlib import Path
from pymatgen.core import Structure

from best_model import BestModel, create_features, get_groups


def read_pymatgen_dict(file):
    with open(file, "r") as f:
        d = json.load(f)
    return Structure.from_dict(d)


def energy_within_threshold(prediction, target):
    # compute absolute error on energy per system.
    # then count the no. of systems where max energy error is < 0.02.
    e_thresh = 0.02
    error_energy = tf.math.abs(target - prediction)

    success = tf.math.count_nonzero(error_energy < e_thresh)
    total = tf.size(target)
    return success / tf.cast(total, tf.int64)


def prepare_dataset(dataset_path):
    dataset_path = Path(dataset_path)
    targets = pd.read_csv(dataset_path / "targets.csv", index_col=0)
    struct = {
        item.name.strip(".json"): read_pymatgen_dict(item)
        for item in (dataset_path / "structures").iterdir()
    }

    data = pd.DataFrame(columns=["structures"], index=struct.keys())
    data = data.assign(structures=struct.values(), targets=targets)

    return data


def main(config):
    train = prepare_dataset(config["datapath"])
    COORDS = train.loc['6141cf13b842c2e72e2f2d4c'].structures.cart_coords
    COORDS = set([tuple(i) for i in COORDS])
    pickle.dump(COORDS, open(config['coords_path'], 'wb'))

    Xtrain = create_features(train, COORDS)
    model = BestModel(num_iterations=5000)
    model.fit(get_groups(Xtrain), train.targets)
    pickle.dump(model, open(config['model_path'], 'wb'))


if __name__ == "__main__":
    with open("config.yaml") as file:
        config = yaml.safe_load(file)
    main(config)
