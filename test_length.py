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
#os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

def test(config, task, model_pt, test_type='test'):
    print(config.device)
    test_set = GoTermDataset(test_type, task)
    test_loader = DataLoader(test_set, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn)
    output_dim = test_set.y_true.shape[-1]
    model = CL_protNET(output_dim, config.esmembed, config.pooling, num_supernode=args.num_node, device_model=config.device, args=args).to(config.device)
    model.load_state_dict(torch.load(model_pt,map_location=config.device))
    model.eval()
    bce_loss = torch.nn.BCELoss()
    
    y_pred_all = []

    y_true_all = test_set.y_true.float()
    with torch.no_grad():
        
        for idx_batch, batch in enumerate(test_loader):
            #model.eval()
            #y_pred, _, _ = model(batch[0].to(config.device))
            y_pred = model(batch[0].to(config.device), batch[2].to(config.device), batch[3].to(config.device), batch[4])
            y_pred_all.append(y_pred)
        y_pred_all = torch.cat(y_pred_all, dim=0).cpu()

        eval_loss = bce_loss(y_pred_all, y_true_all)
        
        Fmax = fmax(y_true_all.numpy(), y_pred_all.numpy())
        aupr = metrics.average_precision_score(y_true_all.numpy(), y_pred_all.numpy(), average='macro')
        Smin = smin(y_true_all.numpy(), y_pred_all.numpy())
        log(f"Test ||| loss: {round(float(eval_loss),3)} ||| aupr: {round(float(aupr),3)} ||| Fmax: {round(float(Fmax),3)} ||| smin: {round(float(Smin),4)}" )
    
    if test_type == 'AF2test':
        result_name = config.test_result_path + 'AF2'+ model_pt.split('/')[-1][6:]
    else:
        result_name = config.test_result_path + model_pt.split('/')[-1][6:]
    with open(result_name, "wb") as f:
        pkl.dump([y_pred_all.numpy(), y_true_all.numpy()], f)

def str2bool(v):
    if isinstance(v,bool):
        return v
    if v == 'True' or v == 'true':
        return True
    if v == 'False' or v == 'false':
        return False

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--task', type=str, default='bp', choices=['bp','mf','cc'], help='')
    p.add_argument('--device', type=str, default='', help='')
    p.add_argument('--model', type=str, default='', help='')
    p.add_argument('--esmembed', default=True, type=str2bool, help='')
    p.add_argument('--pooling', default='MTP', type=str, choices=['MTP','GMP'], help='Multi-set transformer pooling or Global mean pooling')
    p.add_argument('--AF2test', default=False, type=str2bool, help='')
    p.add_argument('--num_node', type=int, default=32, help='')
    p.add_argument('--ista_step_size', type=float, default=0.01, help='')
    p.add_argument('--ista_lambd', type=float, default=0.001, help='')
    p.add_argument('--eigenvec_ratio', type=float, default=0.1, help='')
    
    
    args = p.parse_args()
    print(args)
    config = get_config()
    config.batch_size = 16 
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    if args.device != '':
        config.device = "cuda:" + args.device
    config.esmembed = args.esmembed
    config.pooling = args.pooling
    
    #folder = f"new_Laplacian_rc_sn{args.num_node//4}_bias0_GSI_GCN_3_is_{args.ista_step_size}_sl_{args.ista_lambd}"
    #folder = f"new_Laplacian_rc_sn{args.num_node//4}_GSI_GCN_3_is_{args.ista_step_size}_sl_{args.ista_lambd}"
    
    #folder = f"new_Laplacian_rc_sn{args.num_node//4}_GSI_GCN_3_is_{args.ista_step_size}_sl_{args.ista_lambd}_evec_{args.eigenvec_ratio}"
    #folder = f"High_eliminate_Laplacian_rc_sn{args.num_node//4}_GSI_GCN_3_is_{args.ista_step_size}_sl_{args.ista_lambd}_evec_{args.eigenvec_ratio}"
    #folder = f"fifth_Low_eliminate_Laplacian_rc_sn{args.num_node//4}_GSI_GCN_3_is_{args.ista_step_size}_sl_{args.ista_lambd}_evec_{args.eigenvec_ratio}"
    folder = f"third_Low_eliminate_Laplacian_rc_sn{args.num_node//4}_GSI_GCN_3_is_{args.ista_step_size}_sl_{args.ista_lambd}_evec_{args.eigenvec_ratio}"
    #folder = f"third_High_eliminate_Laplacian_rc_sn{args.num_node//4}_GSI_GCN_3_is_{args.ista_step_size}_sl_{args.ista_lambd}_evec_{args.eigenvec_ratio}"
    #folder = f"fifth_High_eliminate_Laplacian_rc_sn{args.num_node//4}_GSI_GCN_3_is_{args.ista_step_size}_sl_{args.ista_lambd}_evec_{args.eigenvec_ratio}"
    #folder = f"High_eliminate_Laplacian_rc_sn{args.num_node//4}_GSI_GCN_5_is_{args.ista_step_size}_sl_{args.ista_lambd}_evec_{args.eigenvec_ratio}"
    #folder = f"Low_eliminate_Laplacian_rc_sn{args.num_node//4}_GSI_GCN_3_is_{args.ista_step_size}_sl_{args.ista_lambd}_evec_{args.eigenvec_ratio}"
    #folder = f"twice_High_eliminate_Laplacian_rc_sn{args.num_node//4}_GSI_GCN_3_is_{args.ista_step_size}_sl_{args.ista_lambd}_evec_{args.eigenvec_ratio}"
    #folder = f"twice_Low_eliminate_Laplacian_rc_sn{args.num_node//4}_GSI_GCN_3_is_{args.ista_step_size}_sl_{args.ista_lambd}_evec_{args.eigenvec_ratio}"
    #folder = f"twice_High_eliminate_Laplacian_rc_sn{args.num_node//4}_GSI_GCN_3_is_{args.ista_step_size}_sl_{args.ista_lambd}_evec_{args.eigenvec_ratio}"
    #folder = f"new_Laplacian_rc_sn{args.num_node//4}_bias0_GSI_GCN_3_is_{args.ista_step_size}_sl_{args.ista_lambd}_evec_{args.eigenvec_ratio}"
    #folder = f"Prenorm_new_Laplacian_rc_sn{args.num_node//4}_bias0_GSI_GCN_3_is_{args.ista_step_size}_sl_{args.ista_lambd}_evec_{args.eigenvec_ratio}"
    
    args.model = folder + f"/model_{args.task}CLaf.pt"
    if not os.path.exists(folder):
        os.makedirs(folder)
    config.model_save_path = folder + "/model_"
    config.loss_save_path = folder + "/loss_"
    config.test_result_path = folder + "/test_"
    
    if not args.AF2test:
        test(config, args.task, args.model)
    else:
        test(config, args.task, args.model, 'AF2test')
