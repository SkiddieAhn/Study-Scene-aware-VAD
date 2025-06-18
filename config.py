from glob import glob
import os


os.makedirs('weights', exist_ok=True)
os.makedirs('results', exist_ok=True)

share_config = {'mode': 'training',
                'dataset': 'shanghai',
                'img_size': (224, 224),
                'data_root': '/home/sha/datasets/'}  # remember the final '/'


class dict2class:
    def __init__(self, config):
        for k, v in config.items():
            self.__setattr__(k, v)

    def print_cfg(self):
        print('\n' + '-' * 30 + f'{self.mode} cfg' + '-' * 30)
        for k, v in vars(self).items():
            print(f'{k}: {v}')
        print()


def update_config(args=None, mode=None):
    share_config['mode'] = mode
    share_config['dataset'] = args.dataset
    share_config['train_data'] = share_config['data_root'] + args.dataset + '/training/'
    share_config['test_data'] = share_config['data_root'] + args.dataset + '/testing/'
    share_config['weight_path'] = 'weights'
    share_config['result_path'] = 'results'
    share_config['embed_dim'] = 768

    if mode != 'save':
        share_config['scene_length'] = args.scene_length
        share_config['clip_length'] = args.clip_length
        share_config['batch_size'] = args.scene_length

    else:
        share_config['save_mode'] = args.save_mode
        share_config['clip_length'] = args.clip_length

    if mode == 'train':
        share_config['wandb'] = args.wandb
        share_config['lr'] = 0.0001
        share_config['iters'] = args.iters
        share_config['resume'] = glob(f'weights/{args.resume}*')[0] if args.resume else None
        share_config['val_interval'] = args.val_interval
        share_config['save_interval'] = args.save_interval
        share_config['manualseed'] = args.manualseed
        share_config['training_mode'] = args.training_mode
        share_config['video_length'] = args.video_length

    elif mode == 'test':
        share_config['trained_model'] = args.trained_model
        share_config['visualization'] = args.visualization

    return dict2class(share_config)  # change dict keys to class attributes