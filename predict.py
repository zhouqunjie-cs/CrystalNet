import os

from crystalnet.parsing import parse_predict_args, modify_predict_args
from crystalnet.train import make_predictions

if __name__ == '__main__':
    args = parse_predict_args()
    test_name, test_prediction = make_predictions(args)

    output_dir = os.path.join(args.test_path, f'seed_{args.seed}')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'predict_fold8_cgcmpnn.csv')

    with open(output_path, 'w') as fw:
        fw.write(f'name,{args.dataset_name}\n')

        for name, prediction in zip(test_name, test_prediction):
            fw.write(f'{name},{",".join([str(predict) for predict in prediction])}\n')
