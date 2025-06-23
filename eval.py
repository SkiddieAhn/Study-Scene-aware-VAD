import torch 
import argparse
from util import str2bool
from eval_folder.train_eval import val_train_eval
from eval_folder.test_eval import val_test_eval
from config import update_config
from model.supconAE import SubconAE

parser = argparse.ArgumentParser(description='Contrastive_learning')
parser.add_argument('--dataset', default='shanghai-sd', type=str, help='The name of the dataset to evaluate.')
parser.add_argument('--trained_model', default=None, type=str, help='The pre-trained model to evaluate.')
parser.add_argument('--visualization', default=False, type=str2bool, nargs='?', const=True, help='video anomaly score visualization')
parser.add_argument('--scene_length', default=4, type=int, help='number of scenes in dataset')
parser.add_argument('--clip_length', default=16, type=int, help='number of frames')


def val(cfg, model=None, iter=None, device=None):
    '''
    ========================================
    This is for evaluation during training.    
    ========================================
    '''
    if model:
        model.eval()
        auc = val_train_eval(cfg, model, iter+1, device)
        model.train()
        return auc 


    '''
    ========================================
    This is for evaluation during testing.    
    ========================================
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 
    model = SubconAE(input_dim=cfg.embed_dim, hidden_dim=256, latent_dim=256, num_classes=cfg.scene_length).to(device)
    model.eval()

    if cfg.trained_model:
        model.load_state_dict(torch.load(f'weights/' + cfg.trained_model + '.pth')['model'])
        iter = torch.load(f'{cfg.weight_path}/' + cfg.trained_model + '.pth')['iter']
        val_test_eval(cfg, model, iter, device)
    else:
        print('no trained model!')


if __name__ == '__main__':
    args = parser.parse_args()
    test_cfg = update_config(args, mode='test')
    test_cfg.print_cfg()
    val(test_cfg)
