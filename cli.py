import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '-d',
    '--dataset',
    type=str,
    required=True,
    help='Dataset name.'
)
parser.add_argument(
    '-m',
    '--model',
    type=str,
    required=True,
    help='Model name.'
)
parser.add_argument(
    '-p',
    '--path',
    type=str,
    required=False,
    default=None,
    help='filepath of the state dicts.'
)
parser.add_argument(
    '-t',
    '--tag',
    type=str,
    required=True,
    help='Experiment tag name.'
)
cli_args = parser.parse_args()


model_name = cli_args.model
dataset_name = cli_args.dataset
dict_path = cli_args.path
tag = f'{cli_args.model}_{cli_args.dataset}--{cli_args.tag}' if cli_args.dict_path else f'checkpoint_{cli_args.model}_{cli_args.dataset}--{cli_args.tag}' 
