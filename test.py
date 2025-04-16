import torch
import numpy as np
from options import *
from models.models import *
from torch.utils.data import DataLoader
from dataloader import *


def test(model):
    model.eval()

    torch.set_float32_matmul_precision('high') 

    test_loader = DataLoader(dataset=Dataset(opt.test_path, file_name='datas.npy'), batch_size=256, shuffle=False, num_workers=0, pin_memory=True)
    model = nn.DataParallel(model)
    model = model.to(opt.device)
    ckp = torch.load(opt.snapshots_folder + 'best_model.pt', map_location=opt.device, weights_only=True)
    model.load_state_dict(ckp)
    
    preds_m, trues_m, preds_e, trues_e, preds_d, trues_d, preds_t, trues_t = [], [], [], [], [], [], [], []
    preds_p, trues_p, preds_s, trues_s = [], [], [], []
    datas = []

    with torch.no_grad():
        for iteration, (x_in, y_m, y_e, y_d, y_t, y_p, y_s) in enumerate(test_loader):
            x_in2 = x_in.clone() 
            x_in = x_in.to(opt.device)

            x_m, x_e, x_d, x_t, x_p, x_s = model(x_in)

            batch,_ = x_m.shape
            for i in range(batch):
                preds_m.append(x_m[i].cpu().detach().numpy().item())
                trues_m.append(y_m[i].cpu().detach().numpy().item())
                preds_e.append(x_e[i].cpu().detach().numpy().item())
                trues_e.append(y_e[i].cpu().detach().numpy().item())
                preds_d.append(x_d[i].cpu().detach().numpy().item())
                trues_d.append(y_d[i].cpu().detach().numpy().item())
                preds_t.append(x_t[i].cpu().detach().numpy().item())
                trues_t.append(y_t[i].cpu().detach().numpy().item())
                preds_p.append(x_p[i].cpu().detach().numpy())
                trues_p.append(y_p[i].cpu().detach().numpy())
                preds_s.append(x_s[i].cpu().detach().numpy())
                trues_s.append(y_s[i].cpu().detach().numpy())
                datas.append(x_in2[i].cpu().detach().numpy())
            
            print(f'\rstep :{iteration + 1}/{len(test_loader)}', end='', flush=True)

    np.save(results_path + 'preds_m.npy', preds_m)
    np.save(results_path + 'trues_m.npy', trues_m)
    np.save(results_path + 'preds_e.npy', preds_e)
    np.save(results_path + 'trues_e.npy', trues_e)
    np.save(results_path + 'preds_d.npy', preds_d)
    np.save(results_path + 'trues_d.npy', trues_d)
    np.save(results_path + 'preds_t.npy', preds_t)
    np.save(results_path + 'trues_t.npy', trues_t)
    np.save(results_path + 'preds_p.npy', preds_p)
    np.save(results_path + 'trues_p.npy', trues_p)
    np.save(results_path + 'preds_s.npy', preds_s)
    np.save(results_path + 'trues_s.npy', trues_s)
    np.save(results_path + 'datas.npy', datas)
 
model = UNet()
# model = model.to(opt.device)

results_path = './results/'
if not os.path.exists(results_path):
    os.mkdir(results_path)
test(model)
print('\n')
