from argparse import Namespace
import os
import pickle

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from pymatgen.io.vasp import Poscar

from .predict import predict
from chemprop.data import CrystalDataset, CrystalDatapoint
from chemprop.utils import load_args, load_checkpoint, load_scalers
from chemprop.features import AtomCustomJSONInitializer, GaussianDistance, load_radius_dict


def make_predictions(args: Namespace):
    """
    Makes predictions. If smiles is provided, makes predictions on smiles. Otherwise makes predictions on args.test_data.

    :param args: Arguments.
    :return: A list of lists of target name and predictions.
    """
    if args.gpu is not None:
        torch.cuda.set_device(args.gpu)

    print('Loading training args')
    scaler, features_scaler = load_scalers(args.checkpoint_paths[0])
    train_args = load_args(args.checkpoint_paths[0])

    # Update args with training arguments
    for key, value in vars(train_args).items():
        if not hasattr(args, key):
            setattr(args, key, value)

    # Load feature object
    ari = AtomCustomJSONInitializer(f'{args.data_path}/atom_init.json')
    dmin, dmax, step, var = args.rbf_parameters
    gdf = GaussianDistance(dmin=dmin, dmax=dmax, step=step, var=var)
    radius_dic = load_radius_dict(f'{args.data_path}/hubbard_u.yaml')
    if os.path.exists(f'{args.test_path}/graph_cache.pickle'):
        with open(f'{args.test_path}/graph_cache.pickle', 'rb') as f:
            test_graph_dict = pickle.load(f)
    else:
        assert "There is no poscar graph cache, please use preprocess.py to generate poscar graph cache!"

    test_name = list(sorted(test_graph_dict.keys(), key=lambda x: int(x.split('-')[1])))
    # test_name = ['cod-2002478-Ba2Cu1Br1O2.poscar', 'scod-2002478-Ba2Cu1Br1O2.poscar']

    print('Load data')
    test_data = CrystalDataset([CrystalDatapoint(crystal_name=name,
                                                 crystal_dict=test_graph_dict[name],
                                                 targets=[0],
                                                 ari=ari,
                                                 gdf=gdf,
                                                 radius_dic=radius_dic,
                                                 args=args) for name in tqdm(test_name)])

    print(f'Test size = {len(test_data):,}')

    # Predict with each model individually and sum predictions
    test_preds = {name: list() for name in test_name}

    print(f'Predicting with an ensemble of {len(args.checkpoint_paths)} models')
    for checkpoint_path in tqdm(args.checkpoint_paths, total=len(args.checkpoint_paths)):
        # Load model
        fold_num = checkpoint_path.split('/')[-3].split('_')[-1]
        model = load_checkpoint(checkpoint_path, cuda=args.cuda)
        model_preds = predict(
            model=model,
            data=test_data,
            batch_size=args.batch_size,
            scaler=scaler
        )

        for name, prediction in zip(test_name, np.array(model_preds)):
            test_preds[name].extend(prediction)

    return test_preds





