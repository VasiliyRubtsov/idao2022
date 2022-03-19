import numpy as np
import pandas as pd


def get_groups(df: pd.DataFrame):
    return df['num_s_changes'] * 1000 + df['num_s_gaps'] * 100 + \
        df['num_mo_changes'] * 10 + df['num_mo_gaps']


def get_structure_features(structures, COORDS):
    coordintes = {
        's_changes': [],
        'mo_changes': [],
        's_gaps': [],
        'mo_gaps': [],
    }
    for atomic_number, coord in zip(structures.atomic_numbers, structures.cart_coords):
        if atomic_number == 34:
            coordintes['s_changes'].append(coord)
        elif atomic_number == 74:
            coordintes['mo_changes'].append(coord)

    coords = set([tuple(i) for i in structures.cart_coords])
    gaps_coords = list(COORDS.difference(coords))

    for gap_coords in gaps_coords:
        if gap_coords[2] > 3 and gap_coords[2] < 4:
            coordintes['mo_gaps'].append(gap_coords)
        else:
            coordintes['s_gaps'].append(gap_coords)

    features = {}

    if len(coordintes['s_changes']) > 0:
        features['s_change_1_x'] = coordintes['s_changes'][0][0]
        features['s_change_1_y'] = coordintes['s_changes'][0][1]
        features['s_change_1_z'] = coordintes['s_changes'][0][2]

    if len(coordintes['s_changes']) > 1:
        features['s_change_2_x'] = coordintes['s_changes'][1][0]
        features['s_change_2_y'] = coordintes['s_changes'][1][1]
        features['s_change_2_z'] = coordintes['s_changes'][1][2]

    if len(coordintes['mo_changes']) > 0:
        features['mo_change_x'] = coordintes['mo_changes'][0][0]
        features['mo_change_y'] = coordintes['mo_changes'][0][1]
        features['mo_change_z'] = coordintes['mo_changes'][0][2]

    if len(coordintes['s_gaps']) > 0:
        features['s_gap_1_x'] = coordintes['s_gaps'][0][0]
        features['s_gap_1_y'] = coordintes['s_gaps'][0][1]
        features['s_gap_1_z'] = coordintes['s_gaps'][0][2]

    if len(coordintes['s_gaps']) > 1:
        features['s_gap_2_x'] = coordintes['s_gaps'][1][0]
        features['s_gap_2_y'] = coordintes['s_gaps'][1][1]
        features['s_gap_2_z'] = coordintes['s_gaps'][1][2]

    if len(coordintes['mo_gaps']) > 0:
        features['mo_gap_x'] = coordintes['mo_gaps'][0][0]
        features['mo_gap_y'] = coordintes['mo_gaps'][0][1]
        features['mo_gap_z'] = coordintes['mo_gaps'][0][2]

    features['num_s_changes'] = len(coordintes['s_changes'])
    features['num_mo_changes'] = len(coordintes['mo_changes'])
    features['num_s_gaps'] = len(coordintes['s_gaps'])
    features['num_mo_gaps'] = len(coordintes['mo_gaps'])

    return features


def get_num_sites(structures):
    return pd.Series([
        i.species_string for i in structures.sites
    ]).value_counts().to_dict()


def get_num_sites(structures):
    return pd.Series([
        i.species_string for i in structures.sites
    ]).value_counts().to_dict()


def get_mask(X, num_s_changes, num_s_gaps, num_mo_changes, num_mo_gaps):
    mask = X.num_s_changes == num_s_changes
    mask = mask & (X.num_s_gaps == num_s_gaps)
    mask = mask & (X.num_mo_changes == num_mo_changes)
    mask = mask & (X.num_mo_gaps == num_mo_gaps)
    return mask


def get_dist(X, x1, x2, y1, y2, z1, z2):
    dist = (X[x1] - X[x2]) ** 2
    dist = dist + (X[y1] - X[y2]) ** 2
    dist = dist + (X[z1] - X[z2]) ** 2
    dist = (dist ** (1 / 2)).apply(int)
    return dist


def create_features(df, COORDS):
    X = pd.DataFrame()

    features = df.structures.apply(lambda x: get_structure_features(x, COORDS))

    for feature in [
        'num_s_changes', 'num_s_gaps', 'num_mo_changes', 'num_mo_gaps',
        's_change_1_x', 's_change_1_y', 's_change_1_z', 's_change_2_x',
        's_change_2_y', 's_change_2_z', 's_gap_1_x', 's_gap_1_y', 's_gap_1_z',
        's_gap_2_x', 's_gap_2_y', 's_gap_2_z', 'mo_gap_x', 'mo_gap_y',
        'mo_gap_z', 'mo_change_x', 'mo_change_y', 'mo_change_z'
    ]:
        X[feature] = features.apply(lambda x: x.get(feature, -100))

    X['dist'] = 0

    mask = get_mask(X, 1, 1, 0, 1)
    X.loc[mask, 'dist'] = get_dist(
        X[mask], 's_gap_1_x', 'mo_gap_x', 's_gap_1_y',
        'mo_gap_y', 's_gap_1_z', 'mo_gap_z',
    )

    mask = get_mask(X, 0, 2, 0, 1)
    X.loc[mask, 'dist'] = np.min([
        get_dist(
            X[mask], 's_gap_1_x', 'mo_gap_x', 's_gap_1_y',
            'mo_gap_y', 's_gap_1_z', 'mo_gap_z',
        ),
        get_dist(
            X[mask], 's_gap_2_x', 'mo_gap_x', 's_gap_2_y',
            'mo_gap_y', 's_gap_2_z', 'mo_gap_z',
        )
    ], axis=0)

    mask = get_mask(X, 2, 0, 0, 1)
    X.loc[mask, 'dist'] = np.min([
        get_dist(
            X[mask], 's_change_1_x', 'mo_gap_x', 's_change_1_y',
            'mo_gap_y', 's_change_1_z', 'mo_gap_z',
        ),
        get_dist(
            X[mask], 's_change_2_x', 'mo_gap_x', 's_change_2_y',
            'mo_gap_y', 's_change_2_z', 'mo_gap_z',
        )
    ], axis=0)

    mask = get_mask(X, 0, 2, 1, 0)
    X.loc[mask, 'dist'] = get_dist(
        X[mask], 's_gap_1_x', 's_gap_2_x', 's_gap_1_y',
        's_gap_2_y', 's_gap_1_z', 's_gap_2_z',
    )
    return X


def get_groups(df: pd.DataFrame):
    return df['dist'] * 10000 + df['num_s_changes'] * 1000 + df['num_s_gaps'] * 100 + \
        df['num_mo_changes'] * 10 + df['num_mo_gaps']


class BestModel:

    def __init__(self, num_iterations=5000, min_target=0, max_target=1.81, eps=0.02):
        self.weights: pd.Series = None
        self.num_iterations = num_iterations
        self.min_target = min_target
        self.max_target = max_target
        self.eps = eps
        self.fill_na_weight = None

    def get_best_prediction(self, y: pd.Series):
        best_error = 2
        best_prediction = None
        for pred in np.linspace(self.min_target, self.max_target, self.num_iterations):
            error = ((y - pred).abs() > 0.02).mean()
            if error < best_error:
                best_error = error
                best_prediction = pred
        return best_prediction

    def fit(self, groups: pd.Series, targets: pd.Series):
        self.weights = targets.groupby(groups).apply(self.get_best_prediction)
        self.fill_na_weight = self.get_best_prediction(targets)

    def predict(self, groups: pd.Series):
        return groups.map(self.weights).fillna(self.fill_na_weight)
