from graph_data import GoTermDataset, collate_fn
from torch.utils.data import DataLoader
from network import CL_protNET
import torch
from sklearn import metrics
from utils import log, PR_metrics, fmax, smin
import argparse
import pickle as pkl
from config import get_config
import numpy as np
from joblib import Parallel, delayed
import os
import evaluation_metrics
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

with open("data/ic_count.pkl", 'rb') as f:
    ic_count = pkl.load(f)
ic_count['bp'] = np.where(ic_count['bp'] == 0, 1, ic_count['bp'])
ic_count['mf'] = np.where(ic_count['mf'] == 0, 1, ic_count['mf'])
ic_count['cc'] = np.where(ic_count['cc'] == 0, 1, ic_count['cc'])
train_ic = {}
train_ic['bp'] = -np.log2(ic_count['bp'] / 69709)
train_ic['mf'] = -np.log2(ic_count['mf'] / 69709)
train_ic['cc'] = -np.log2(ic_count['cc'] / 69709)

def test(config, model_pt, test_type='test'):
    print(config.device)
    test_set = GoTermDataset(test_type)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    output_dim = test_set.y_true.shape[-1]
    model = CL_protNET(output_dim, config.esmembed, num_supernode=args.num_node, device_model=config.device, args=args).to(config.device)
    model.load_state_dict(torch.load(model_pt,map_location=config.device))
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model.eval()
    bce_loss = torch.nn.BCELoss()
    
    y_pred_all_bp = []
    y_pred_all_mf = []
    y_pred_all_cc = []

    y_true_all = test_set.y_true.float()
    y_true_all_at = torch.split(test_set.y_true.float(), [1943, 489, 320], dim=1)
    y_true_all_bp = y_true_all_at[0]
    y_true_all_mf = y_true_all_at[1]
    y_true_all_cc = y_true_all_at[2]

    with torch.no_grad():
        for idx_batch, batch in enumerate(test_loader):
            y_pred_bp, y_pred_mf, y_pred_cc, _ = model(batch[0].to(config.device), batch[2].to(config.device), batch[3].to(config.device), batch[4])
            y_pred_all_bp.append(y_pred_bp)
            y_pred_all_mf.append(y_pred_mf)
            y_pred_all_cc.append(y_pred_cc)
        y_pred_all_bp = torch.cat(y_pred_all_bp, dim=0).cpu()
        y_pred_all_mf = torch.cat(y_pred_all_mf, dim=0).cpu()
        y_pred_all_cc = torch.cat(y_pred_all_cc, dim=0).cpu()

        eval_loss = bce_loss(y_true_all_bp, y_pred_all_bp)
        eval_loss += bce_loss(y_true_all_mf, y_pred_all_mf)
        eval_loss += bce_loss(y_true_all_cc, y_pred_all_cc)
        
        Fmax_bp = evaluation_metrics.fmax(y_true_all_bp.numpy(), y_pred_all_bp.numpy(), "bp")
        Fmax_mf = evaluation_metrics.fmax(y_true_all_mf.numpy(), y_pred_all_mf.numpy(), "mf")
        Fmax_cc = evaluation_metrics.fmax(y_true_all_cc.numpy(), y_pred_all_cc.numpy(), "cc")
        
        aupr_bp = evaluation_metrics.macro_aupr_test(y_true_all_bp.numpy(), y_pred_all_bp.numpy())
        aupr_mf = evaluation_metrics.macro_aupr_test(y_true_all_mf.numpy(), y_pred_all_mf.numpy())
        aupr_cc = evaluation_metrics.macro_aupr_test(y_true_all_cc.numpy(), y_pred_all_cc.numpy())

        Smin_bp = evaluation_metrics.smin(train_ic["bp"], y_true_all_bp.numpy(), y_pred_all_bp.numpy())
        Smin_mf = evaluation_metrics.smin(train_ic["mf"], y_true_all_mf.numpy(), y_pred_all_mf.numpy())
        Smin_cc = evaluation_metrics.smin(train_ic["cc"], y_true_all_cc.numpy(), y_pred_all_cc.numpy())
        log(f"Test ||| loss: {round(float(eval_loss),3)}" )
        log(f"Test ||| aupr_bp: {round(float(aupr_bp),3)} ||| Fmax_bp: {round(float(Fmax_bp),3)} ||| smin_bp: {round(float(Smin_bp),4)}" )
        log(f"Test ||| aupr_mf: {round(float(aupr_mf),3)} ||| Fmax_mf: {round(float(Fmax_mf),3)} ||| smin_mf: {round(float(Smin_mf),4)}" )
        log(f"Test ||| aupr_cc: {round(float(aupr_cc),3)} ||| Fmax_cc: {round(float(Fmax_cc),3)} ||| smin_cc: {round(float(Smin_cc),4)}" )
        log(f"Learnable Parameters: {num_params}")

    #if test_type == 'AF2test':
    #    result_name = config.test_result_path + 'AF2'+ model_pt.split('/')[-1][6:]
    #else:
    #    result_name = config.test_result_path + model_pt.split('/')[-1][6:]
    #with open(result_name, "wb") as f:
    #    pkl.dump([y_pred_all.numpy(), y_true_all.numpy()], f)

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
    p.add_argument('--model', type=str, default='', help='')
    p.add_argument('--esmembed', default=True, type=str2bool, help='')
    p.add_argument('--AF2test', default=False, type=str2bool, help='')
    p.add_argument('--num_node', type=int, default=32, help='')
    p.add_argument('--eigenvec_ratio', type=float, default=0.15, help='')
    p.add_argument('--num_head', type=int, default=4, help='')
    p.add_argument('--temperature', type=float, default=0.1, help='')
    
    args = p.parse_args()
    print(args)
    config = get_config()
    config.batch_size = 16 
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if args.device != '':
        config.device = "cuda:" + args.device
    config.esmembed = args.esmembed
    
    folder = f"S2RL_snode_{args.num_node}_evec_{args.eigenvec_ratio}_head_{args.num_head}_temp_{args.temperature}"
    
    args.model = folder + f"/model_multitaskCLaf.pt"
    if not os.path.exists(folder):
        os.makedirs(folder)
    config.model_save_path = folder + "/model_"
    config.loss_save_path = folder + "/loss_"
    config.test_result_path = folder + "/test_"
    
    if not args.AF2test:
        test(config, args.model)
    else:
        test(config, args.model, 'AF2test')
