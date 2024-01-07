import tomli
import shutil
import os
import argparse
from tab_ddpm.util import try_argparse
from tab_ddpm.scripts.train import train as _train
from tab_ddpm.scripts.sample import sample as _sample
import delu as zero
from tab_ddpm import lib
import torch

DEFAULT_DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")

def load_config(path) :
    with open(path, 'rb') as f:
        return tomli.load(f)
    
def save_file(parent_dir, config_path):
    try:
        dst = os.path.join(parent_dir)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(os.path.abspath(config_path), dst)
    except shutil.SameFileError:
        pass

def main(
    config=None,
    train=False,
    sample=False,
    eval=False,
    change_val=False
):
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', metavar='FILE', default=config)
    parser.add_argument('--train', action='store_true', default=train)
    parser.add_argument('--sample', action='store_true',  default=sample)
    parser.add_argument('--eval', action='store_true',  default=eval)
    parser.add_argument('--change_val', action='store_true',  default=change_val)

    args = try_argparse(parser)
    assert args.config

    raw_config = lib.load_config(args.config)
    if 'device' in raw_config and torch.cuda.is_available():
        device = torch.device(raw_config['device'] if torch.cuda.is_available() else "cpu")
    else:
        device = DEFAULT_DEVICE
    
    timer = zero.Timer()
    timer.run()
    save_file(os.path.join(raw_config['parent_dir'], 'config.toml'), args.config)

    if args.train:
        _train(
            **raw_config['train']['main'],
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            change_val=args.change_val
        )
    if args.sample:
        _sample(
            num_samples=raw_config['sample']['num_samples'],
            batch_size=raw_config['sample']['batch_size'],
            disbalance=raw_config['sample'].get('disbalance', None),
            **raw_config['diffusion_params'],
            parent_dir=raw_config['parent_dir'],
            real_data_path=raw_config['real_data_path'],
            model_path=os.path.join(raw_config['parent_dir'], 'model.pt'),
            model_type=raw_config['model_type'],
            model_params=raw_config['model_params'],
            T_dict=raw_config['train']['T'],
            num_numerical_features=raw_config['num_numerical_features'],
            device=device,
            seed=raw_config['sample'].get('seed', 0),
            change_val=args.change_val
        )

    save_file(os.path.join(raw_config['parent_dir'], 'info.json'), os.path.join(raw_config['real_data_path'], 'info.json'))
    # if args.eval:
    #     if raw_config['eval']['type']['eval_model'] == 'catboost':
    #         train_catboost(
    #             parent_dir=raw_config['parent_dir'],
    #             real_data_path=raw_config['real_data_path'],
    #             eval_type=raw_config['eval']['type']['eval_type'],
    #             T_dict=raw_config['eval']['T'],
    #             seed=raw_config['seed'],
    #             change_val=args.change_val
    #         )
    #     elif raw_config['eval']['type']['eval_model'] == 'mlp':
    #         train_mlp(
    #             parent_dir=raw_config['parent_dir'],
    #             real_data_path=raw_config['real_data_path'],
    #             eval_type=raw_config['eval']['type']['eval_type'],
    #             T_dict=raw_config['eval']['T'],
    #             seed=raw_config['seed'],
    #             change_val=args.change_val,
    #             device=device
    #         )
    #     elif raw_config['eval']['type']['eval_model'] == 'simple':
    #         train_simple(
    #             parent_dir=raw_config['parent_dir'],
    #             real_data_path=raw_config['real_data_path'],
    #             eval_type=raw_config['eval']['type']['eval_type'],
    #             T_dict=raw_config['eval']['T'],
    #             seed=raw_config['seed'],
    #             change_val=args.change_val
    #         )

    print(f'Elapsed time: {str(timer)}')

if __name__ == '__main__':
    main()
