import torch 
import matplotlib.pyplot as plt


def make_models_dict(model, opt, iter):
    model_dict = {'model': model.state_dict(), 
                  'optim': opt.state_dict(),
                  'iter': iter}
    return model_dict


def update_best_model(cfg, iter, auc, best_auc, model, opt):
    if best_auc < auc:
        best_auc = auc
        model_dict = make_models_dict(model, opt, iter+1)
        save_path = f"{cfg.weight_path}/best_model_{cfg.dataset}_{cfg.training_mode}_{cfg.manualseed}.pth"
        torch.save(model_dict, save_path)

        print('\n<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        print(f"[best model] update! at {iter+1} iteration!! [auc: {best_auc:.3f}]")
        print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>\n')
    return best_auc