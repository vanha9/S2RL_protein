from graph_data import GoTermDataset, collate_fn
from torch.utils.data import DataLoader
from network import CL_protNET
from nt_xent import NT_Xent
import torch.nn.functional as F
import torch
from torch.autograd import Variable
from sklearn import metrics
from utils import log
import argparse
from config import get_config
import numpy as np
import os
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore")


def train(config, args):

    train_set = GoTermDataset("train", config.AF2model)
    #pos_weights = torch.tensor(train_set.pos_weights).float()
    valid_set = GoTermDataset("val", config.AF2model)
    train_loader = DataLoader(train_set, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    
    
    output_dim = valid_set.y_true.shape[-1]
    model = CL_protNET(output_dim, config.esmembed, num_supernode=args.num_node, device_model=config.device, args=args).to(config.device)
    optimizer = torch.optim.Adam(
        params = model.parameters(), 
        **config.optimizer,
        )
    bce_loss = torch.nn.BCELoss(reduce=False)

    train_loss = []
    val_loss = []
    val_aupr_bp = []
    val_aupr_mf = []
    val_aupr_cc = []
    val_Fmax = []
    es = 0
    
    y_true_all_at = torch.split(valid_set.y_true.float(), [1943, 489, 320], dim=1)
    y_true_all_bp = y_true_all_at[0].reshape(-1)
    y_true_all_mf = y_true_all_at[1].reshape(-1)
    y_true_all_cc = y_true_all_at[2].reshape(-1)

    for ith_epoch in range(config.max_epochs):
        # scheduler.step()
        for idx_batch, batch in enumerate(tqdm(train_loader)):
            model.train()
            y_pred_bp, y_pred_mf, y_pred_cc, decorrelation_loss = model(batch[0].to(config.device), batch[2].to(config.device), batch[3].to(config.device), batch[4])

            y_true = batch[1].to(config.device)
            y_true_at = torch.split(y_true, [1943, 489, 320], dim=1)

            y_true_bp = y_true_at[0].to(config.device)
            _loss_bp = bce_loss(y_pred_bp, y_true_bp)
            _loss_bp = _loss_bp.mean()

            y_true_mf = y_true_at[1].to(config.device)
            _loss_mf = bce_loss(y_pred_mf, y_true_mf)
            _loss_mf = _loss_mf.mean()

            y_true_cc = y_true_at[2].to(config.device)
            _loss_cc = bce_loss(y_pred_cc, y_true_cc)
            _loss_cc = _loss_cc.mean()
                
            loss = _loss_bp + _loss_mf + _loss_cc 
            loss = loss + decorrelation_loss * 1e-6

            train_loss.append(loss.clone().detach().cpu().numpy())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if ith_epoch == 0:
            for name, param in model.named_parameters():
                if param.grad is None:
                    print(f"Parameter '{name}' has a None gradient")
        #        exit()
                
        eval_loss = 0
        model.eval()
        y_pred_all_bp = []
        y_pred_all_mf = []
        y_pred_all_cc = []
        n_nce_all = []
        
        with torch.no_grad():
            for idx_batch, batch in enumerate(val_loader):
                y_pred_bp, y_pred_mf, y_pred_cc, _ = model(batch[0].to(config.device), batch[2].to(config.device), batch[3].to(config.device), batch[4])
                y_pred_all_bp.append(y_pred_bp)
                y_pred_all_mf.append(y_pred_mf)
                y_pred_all_cc.append(y_pred_cc)
            
            y_pred_all_bp = torch.cat(y_pred_all_bp, dim=0).cpu().reshape(-1)
            y_pred_all_mf = torch.cat(y_pred_all_mf, dim=0).cpu().reshape(-1)
            y_pred_all_cc = torch.cat(y_pred_all_cc, dim=0).cpu().reshape(-1)
            
            eval_loss = bce_loss(y_pred_all_bp, y_true_all_bp).mean()
            eval_loss += bce_loss(y_pred_all_mf, y_true_all_mf).mean()
            eval_loss += bce_loss(y_pred_all_cc, y_true_all_cc).mean()

            aupr_bp = metrics.average_precision_score(y_true_all_bp.numpy(), y_pred_all_bp.numpy(), average="samples")
            aupr_mf = metrics.average_precision_score(y_true_all_mf.numpy(), y_pred_all_mf.numpy(), average="samples")
            aupr_cc = metrics.average_precision_score(y_true_all_cc.numpy(), y_pred_all_cc.numpy(), average="samples")
            val_aupr_bp.append(aupr_bp)
            val_aupr_mf.append(aupr_mf)
            val_aupr_cc.append(aupr_cc)
            log(f"{ith_epoch} VAL_epoch ||| loss: {round(float(eval_loss),3)} ||| aupr_bp: {round(float(aupr_bp),3)} ||| aupr_mf: {round(float(aupr_mf),3)} ||| aupr_cc: {round(float(aupr_cc),3)}")
            if ith_epoch == 0:
                best_eval_loss = eval_loss
            if eval_loss < best_eval_loss:
                best_eval_loss = eval_loss
                es = 0
                torch.save(model.state_dict(), config.model_save_path + "multitaskCLaf.pt")
            else:
                es += 1
                print("Counter {} of 5".format(es))

            if es > 4:
                
                torch.save(
                    {
                        "train_bce": train_loss,
                        "val_bce": eval_loss.detach().cpu().numpy(),
                        "val_aupr_bp": val_aupr_bp,
                        "val_aupr_mf": val_aupr_mf,
                        "val_aupr_cc": val_aupr_cc,
                    }, config.loss_save_path + "multitaskCLaf.pt"
                )

                break

def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True' or v == 'true':
        return True
    if v == 'False' or v == 'false':
        return False


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--device', type=str, default='', help='')
    p.add_argument('--esmembed', default=True, type=str2bool, help='')
    p.add_argument('--AF2model', default=True, type=str2bool, help='whether to use AF2model for training')
    p.add_argument('--batch_size', type=int, default=64, help='')
    p.add_argument('--num_node', type=int, default=32, help='')
    p.add_argument('--eigenvec_ratio', type=float, default=0.15, help='')
    p.add_argument('--num_head', type=int, default=4, help='')
    p.add_argument('--temperature', type=float, default=0.1, help='')

    args = p.parse_args()
    config = get_config()
    config.optimizer['lr'] = 1e-4
    config.batch_size = args.batch_size
    config.max_epochs = 100
    if args.device != '':
        config.device = "cuda:" + args.device
    config.esmembed = args.esmembed
    print(args)
    config.AF2model = args.AF2model

    folder = f"S2RL_snode_{args.num_node}_evec_{args.eigenvec_ratio}_head_{args.num_head}_temp_{args.temperature}"
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    config.model_save_path = folder + "/model_"
    config.loss_save_path = folder + "/loss_"
    config.test_result_path = folder + "/test_"
    train(config, args)

