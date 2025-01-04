# -*- encoding: utf-8 -*-
'''
@File    :   main.py
@Time    :   2023/11/07 19:17:31
@Author  :   Fei Gao
'''

from src.dataset import ConcDataset, build_edges
from src.model import HyperMP

import os
import torch
import pickle as  pkl
from tqdm import tqdm
from argparse import ArgumentParser
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

TARGET_MOLECULES = ['NO2', 'NO', 'O3', 'O3P', 'NO3', 'N2O5', 'OH']
with open("./data/processed_data/species_list.pkl", "rb") as f:
    species_list = pkl.load(f)
TARGET_MOLECULES = species_list
TARGET_MOLECULES_INDEX = [species_list.index(m) for m in TARGET_MOLECULES]

def train(model, batch, edges, device, optimizer, iteration, **kwargs):
    writer = kwargs.get("writer", None)
    model.train()
    x, y = [item.to(device) for item in batch]
    optimizer.zero_grad()
    y_hat = model(x, edges)
    loss = model.loss(y, y_hat)
    if writer is not None:
        abs_errors = torch.abs(y - y_hat).detach().cpu()
        # 计算最小值、中值和最大值
        min_error = torch.min(abs_errors).item()
        median_error = torch.median(abs_errors).item()
        max_error = torch.max(abs_errors).item()
        mean_error = torch.mean(abs_errors).item()

        # 记录这些值
        writer.add_scalar('Error/Min', min_error, iteration)
        writer.add_scalar('Error/Median', median_error, iteration)
        writer.add_scalar('Error/Max', max_error, iteration)
        writer.add_scalar('Error/Mean', mean_error, iteration)
        # 记录训练集的损失值到TensorBoard
        writer.add_scalar('Loss/Train', loss, iteration)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
    optimizer.step()
    loss = loss.item()
    print("Iter.: {:<5} | Train (scaled) Loss: {:.2e} | lr: {:.2e}".format(iteration, loss, optimizer.param_groups[0]['lr']))
    
    
    
def val(model, val_dataloader, val_dataset, edges, device):
    model.eval()
    val_loss = []
    val_pred_mse = []
    val_base_mse = []
    with torch.no_grad():
        for batch in val_dataloader:
            # 移动数据到GPU上
            x, y = [item.to(device) for item in batch]
            y_hat = model(x, edges)
            loss = model.loss(y, y_hat)
            val_loss.append(loss.item())
            pred_mse, base_mse = model.conc_mse(x, y, y_hat, val_dataset)
            val_pred_mse.append(pred_mse.item())
            val_base_mse.append(base_mse.item())
    
    average_val_loss = torch.Tensor(val_loss).mean()
    average_val_pred_mse = torch.sqrt(torch.Tensor(val_pred_mse).mean())
    average_val_base_mse = torch.sqrt(torch.Tensor(val_base_mse).mean())
    
    print("Validation Loss: {:.1e},  RMSE: {:.1e}, Base RMSE: {:.1e}".format(average_val_loss, average_val_pred_mse, average_val_base_mse))
    
    return average_val_loss

def helper(val_loss, best_val_loss, patience_counter, iteration, model, optimizer, lr_scheduler, args):
    # 早停。更新学习率调度器
    lr_scheduler.step(torch.Tensor(val_loss).mean())
    # 检查验证损失是否改善以及早停条件
    current_val_loss = torch.Tensor(val_loss).mean().item()
    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        patience_counter = 0
        # 保存检查点
        torch.save({
            'iteration': iteration,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': best_val_loss,}, args.checkpoint_path)

        print(f"Checkpoint saved at iteration {iteration} with val loss: {best_val_loss:.2e}")
    else:
        patience_counter += 1

    # 早停判断
    early_stopping = False
    if patience_counter >= args.early_stopping_patience:
        print(f"Early stopping triggered after {iteration + 1} iterations.")
        early_stopping = True
    
    return best_val_loss, patience_counter, early_stopping
    

def test(model, args, test_dataloader, test_dataset, edges, device):
    print("Loading model checkpoint from {}...".format(args.checkpoint_path))
    checkpoint = torch.load(args.checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    test_loss = []
    test_pred_mse = []
    test_base_mse = []
    print("Testing...")
    pbar = tqdm(test_dataloader)
    pbar.set_description(f"Test")
    with torch.no_grad():
        # store all x and y
        x_all = []
        y_all = []
        y_hat_all = []
        for batch in pbar:
            # 移动数据到GPU上
            x, y = [item.to(device) for item in batch] # [batch_size, number_nodes, 1]
            y_hat = model(x, edges)
            loss = model.loss(y, y_hat)
            test_loss.append(loss.item())
            pred_mse, base_mse = model.conc_mse(x, y, y_hat, test_dataset)
            test_pred_mse.append(pred_mse.item())
            test_base_mse.append(base_mse.item())
            pbar.set_postfix({"(scaled) Loss": "{:.1e}".format(torch.Tensor(test_loss).mean()),
                                "RMSE": "{:.1e}".format(torch.sqrt(torch.Tensor(test_pred_mse).mean())),
                                "Base RMSE": "{:.1e}".format(torch.sqrt(torch.Tensor(test_base_mse).mean()))})
            x_all.append(x.cpu().squeeze())
            y_all.append(y.cpu().squeeze())
            y_hat_all.append(y_hat.cpu().squeeze())
    avg_test_loss = torch.Tensor(test_loss).mean()
    avg_test_pred_mse = torch.sqrt(torch.Tensor(test_pred_mse).mean())
    avg_test_base_mse = torch.sqrt(torch.Tensor(test_base_mse).mean())
    print("Test Loss: {:.2e}, Test RMSE: {:.2e}, Test Base RMSE: {:.2e}".format(avg_test_loss, avg_test_pred_mse, avg_test_base_mse))
    
    x_all = torch.cat(x_all, dim=0) # [number_samples, number_mecules]
    y_all = torch.cat(y_all, dim=0)
    y_hat_all = torch.cat(y_hat_all, dim=0)
    
    # get the R2 and RMSE of target melecules
    model = model.cpu()
    pred_mse, base_mse = model.conc_mse(x_all, y_all, y_hat_all, test_dataset, dim=0)
    pred_r2, base_r2 = model.conc_r2(x_all, y_all, y_hat_all, test_dataset, dim=0)
    
    for i, name in zip(TARGET_MOLECULES_INDEX, TARGET_MOLECULES):
        prmse = torch.sqrt(pred_mse[i])
        brmse = torch.sqrt(base_mse[i])
        pr2 = pred_r2[i]
        br2 = base_r2[i]   
        print("Molecule: {:<5} | Pred RMSE: {:.1e} | Base RMSE: {:.1e} | Pred R2: {:.5f} | Base R2: {:.5f}".format(name, prmse, brmse, pr2, br2))


def main():
    parser = ArgumentParser()
    parser.add_argument("--path2reaction", type=str, default="./data/processed_data/cleaned_reactions.pkl")
    parser.add_argument("--path2species", type=str, default="./data/processed_data/species_list.pkl")
    parser.add_argument("--path2datasets", type=str, default="./data/processed_data/merged_dataset.pkl")
    parser.add_argument("--in_dim", type=int, default=1)
    parser.add_argument("--hid_dim", type=int, default=64)
    parser.add_argument("--out_dim", type=int, default=1)
    parser.add_argument("--num_mp", type=int, default=2,
                        help="Number of message passing layers")
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--min_max_scale", action="store_true", default=False,
                        help="If true, use min-max scaling to scale the data")
    parser.add_argument("--num_iterations", type=int, default=10000, 
                        help="Number of iterations to train the model")
    parser.add_argument("--val_every", type=int, default=100,
                        help="Number of iterations to validate the model")
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--dropout_rate", type=float, default=0.)
    parser.add_argument("--cuda", type=int, default=0)
    parser.add_argument("--early_stopping_patience", type=int, default=7,
                        help="Number of epochs to wait for improvement before stopping the training")
    parser.add_argument("--lr_scheduler_patience", type=int, default=1,
                        help="Number of epochs to wait before reducing the learning rate")
    parser.add_argument("--lr_scheduler_factor", type=float, default=0.1,
                        help="Factor by which the learning rate will be reduced. new_lr = lr * factor")
    parser.add_argument("--checkpoint_path", type=str, default="./checkpoints/",
                        help="Path to save the model checkpoint")
    parser.add_argument("--only_test", action="store_true", default=False,
                        help="If true, only test the model")
    args = parser.parse_args()
    
    writer = None
    if not args.only_test:
        time_now = datetime.now().strftime("%Y%m%d-%H%M%S")
        writer = SummaryWriter(log_dir=f"./logs/{time_now}")
        writer.add_text("args", str(args))
        args.checkpoint_path = os.path.join(args.checkpoint_path, time_now + ".pt")
    else:
        # check if checkpoint file exists
        assert os.path.exists(args.checkpoint_path), "Checkpoint file does not exist!"
    
    # 检查是否有CUDA设备可用，如果有，使用第一个可用的GPU
    device = torch.device("cuda:{}".format(args.cuda) if torch.cuda.is_available() and args.cuda != -1 else "cpu")
    print(f"Using device: {device}")
    
    # load graph and datasets
    print("Loading basic data...")
    with open(args.path2reaction, "rb") as f:
        reactions = pkl.load(f)
    with open(args.path2species, "rb") as f:
        species = pkl.load(f)
    NumberReactions = len(reactions)
    NumberMolecules = len(species)
    
    print("Loading train/val/test datasets...")
    with open(args.path2datasets, "rb") as f:
        all_dataset = pkl.load(f)
    
    print("Building edges...")
    # build edges and train/val/test dataset
    edges = build_edges(reactions, species)
    edges = [e.to(device) for e in edges]
    if not args.only_test:
        train_dataset = ConcDataset(all_dataset, dataset_type="train", min_max_scale=args.min_max_scale)
        val_dataset = ConcDataset(all_dataset, dataset_type="val", min_max_scale=args.min_max_scale)
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    test_dataset = ConcDataset(all_dataset, dataset_type="test", min_max_scale=args.min_max_scale)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    
    # build model
    model = HyperMP(in_dim=args.in_dim,
                    hid_dim=args.hid_dim,
                    out_dim=args.out_dim,
                    num_message_passing=args.num_mp,
                    num_molecules=NumberMolecules,
                    num_reactions=NumberReactions,
                    dropout=args.dropout_rate)
    model.to(device)
    # build optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    # 初始化学习率调度器
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', 
                                                              patience=args.lr_scheduler_patience,
                                                              factor=args.lr_scheduler_factor,
                                                              verbose=True)
    
    if args.only_test:
        # test the model and exit
        test(model, args, test_dataloader, test_dataset, edges, device)
        return

    best_val_loss, patience_counter = float('inf'), 0
    for iteration in range(args.num_iterations):
        batch = next(iter(train_dataloader))
        train(model, batch, edges, device, optimizer, iteration, writer=writer)
        
        if iteration % args.val_every == 0 and iteration != 0:
            print("Validating...")
            val_loss = val(model, val_dataloader, val_dataset, edges, device)
            best_val_loss, patience_counter, early_stopping = helper(val_loss, best_val_loss, patience_counter, iteration, model, optimizer, lr_scheduler, args)
            if early_stopping:
                break
            
    test(model, args, test_dataloader, test_dataset, edges, device)
        
if __name__ == "__main__":
    main()