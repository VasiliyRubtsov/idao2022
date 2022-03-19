from pathlib import Path

import yaml
import pickle
import pandas as pd

from baseline import read_pymatgen_dict
from best_model import create_features, get_groups


def main(config):

    model = pickle.load(open(config['model_path'], 'rb'))
    COORDS = pickle.load(open(config['coords_path'], 'rb'))
    dataset_path = Path(config['test_datapath'])
    struct = {item.name.strip('.json'): read_pymatgen_dict(item) for item in (dataset_path / 'structures').iterdir()}
    private_test = pd.DataFrame(columns=['id', 'structures'], index=struct.keys())
    private_test = private_test.assign(structures=struct.values())
    private_test = private_test.assign(predictions=model.predict(
        get_groups(create_features(private_test, COORDS))
    ))
    private_test[['predictions']].to_csv('./submission.csv', index_label='id')


if __name__ == '__main__':
    with open("config.yaml") as file:
        config = yaml.safe_load(file)
    main(config)
