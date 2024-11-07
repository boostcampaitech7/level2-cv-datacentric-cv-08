import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser
import services.kakao as kakao
import services.spreadsheet as sp
import json
import services.slack as slack
import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm
from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST
import optuna
import mlflow
import mlflow.pytorch
import torch_optimizer as optim

'''
    server_number : 사용중인 서버숫자
    name : 작업자 이름
    status : 학습중, 학습완료
    task : 어떤 작업을 수행하는가
    mlflow url : mlflow 주소
    mlflow experiment : mlflow 실험이름
    mlflow_runname : mlflow 런이름
    n_startup_trials : Optuna 초기 학습 시 무조건 실행하는 시도 횟수
    n_warmup_steps : Optuna Pruner가 모델 학습 중간에서 학습을 중단하기 전에 대기할 단계 수
    interval_steps : Optuna Pruner가 학습 상태를 확인하는 간격
    n_trials : Optuna 최적화에서 수행할 실험 횟수
    optims=사용하고자 하는 최적화 알고리즘 목록, ['Adam', 'AdamW', 'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Nadam'] 
     
'''
server_number = 1
name = "임용섭"
task = "서버 1 optuna 테스트"
mlflow_url = "https://shoe-worker-recommends-boutique.trycloudflare.com"
mlflow_exp = "server1"
mlflow_runname = "run-again"

n_startup_trials=3  
n_warmup_steps=5 
interval_steps=5 
n_trials=10
optims=['AdamW','SGD','Nadam','Radam']

with open("/data/ephemeral/home/key/uuid.json", 'r') as fp:
    receiver_uuids = json.load(fp)
receiver_uuids = list(receiver_uuids.keys())

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR', 'trained_models'))
    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=30)
    parser.add_argument('--save_interval', type=int, default=5)
    
    args = parser.parse_args()
    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')
    return args

def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, optimizer, trial):

    sp.update_server_status(server_number=server_number, name=name, status=True, task=task)
    kakao.send_message(receiver_uuids=receiver_uuids, 
                       message_text=f"{name}님이 서버 {server_number}번에서 {max_epoch}epoch {task}학습을 시작하였습니다.")
    slack.send_slack_notification(f"{name}님이 서버 {server_number}번에서 {max_epoch}epoch {task}학습을 시작하였습니다.")
    mlflow.set_tracking_uri(mlflow_url)
    mlflow.set_experiment(mlflow_exp)
    
    with mlflow.start_run(run_name=mlflow_runname) as run:
        mlflow.log_params({
            "data_dir": data_dir,
            "model_dir": model_dir,
            "device": device,
            "image_size": image_size,
            "input_size": input_size,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_epoch": max_epoch
        })
        
        train_start_time = time.time()
        dataset = SceneTextDataset(data_dir, split='train', image_size=image_size, crop_size=input_size)
        dataset = EASTDataset(dataset)
        num_batches = math.ceil(len(dataset) / batch_size)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        
        model = EAST().to(device)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)
        model.train()

        try:
            for epoch in range(max_epoch):
                epoch_loss, epoch_start = 0, time.time()
                with tqdm(total=num_batches) as pbar:
                    for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                        pbar.set_description(f'[Epoch {epoch + 1}]')
                        loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        epoch_loss += loss.item()
                        pbar.update(1)
                        val_dict = {'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'], 'IoU loss': extra_info['iou_loss']}
                        pbar.set_postfix(val_dict)
                        
                        mlflow.log_metrics(val_dict, step=epoch * num_batches + pbar.n)
                
                avg_loss = epoch_loss / num_batches
                mlflow.log_metric("Mean loss", avg_loss, step=epoch)
                
               
                trial.report(avg_loss, step=epoch)
                
    
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()

                scheduler.step()
                print(f'Mean loss: {avg_loss:.4f} | Elapsed time: {timedelta(seconds=time.time() - epoch_start)}')

                if (epoch + 1) % save_interval == 0:
                    if not osp.exists(model_dir):
                        os.makedirs(model_dir)
                    ckpt_fpath = osp.join(model_dir, 'latest.pth')
                    torch.save(model.state_dict(), ckpt_fpath)
                    mlflow.log_artifact(ckpt_fpath)

            return avg_loss  
        except Exception as e:
            kakao.send_message(receiver_uuids=receiver_uuids, message_text=f"서버 {server_number}번에서 {name}님의 {task} 학습 도중 에러가 발생하였습니다. 확인이 필요합니다.")
            slack.send_slack_notification(f"서버 {server_number}번에서 {name}님의 {task} 학습 도중 에러가 발생하였습니다. {e}")

def objective(trial,optimizer_name):

    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-3)

    
    model = EAST()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    

    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=learning_rate,
            betas=(trial.suggest_float('beta1', 0.85, 0.95), trial.suggest_float('beta2', 0.9, 0.999)),
            eps=trial.suggest_loguniform('eps', 1e-9, 1e-7)
        )
    elif optimizer_name == 'AdamW':
        weight_decay = trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(trial.suggest_float('beta1', 0.85, 0.95), trial.suggest_float('beta2', 0.9, 0.999)),
            eps=trial.suggest_loguniform('eps', 1e-9, 1e-7)
        )
    elif optimizer_name == 'SGD':
        momentum = trial.suggest_float('momentum', 0.5, 0.9)
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            momentum=momentum,
            dampening=0,
        
            weight_decay=trial.suggest_loguniform('weight_decay', 1e-5, 1e-2),
            nesterov=True
        )
    elif optimizer_name == 'RMSprop':
        optimizer = torch.optim.RMSprop(
            model.parameters(),
            lr=learning_rate,
            alpha=trial.suggest_float('alpha', 0.9, 0.99),
            eps=trial.suggest_loguniform('eps', 1e-9, 1e-7),
            weight_decay=trial.suggest_loguniform('weight_decay', 1e-5, 1e-2),
            momentum=trial.suggest_float('momentum', 0.5, 0.9)
        )
    elif optimizer_name == 'Adagrad':
        optimizer = torch.optim.Adagrad(
            model.parameters(),
            lr=learning_rate,
            lr_decay=trial.suggest_loguniform('lr_decay', 1e-5, 1e-3),
            weight_decay=trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
        )
    elif optimizer_name == 'Adadelta':
        optimizer = torch.optim.Adadelta(
            model.parameters(),
            lr=learning_rate,
            rho=trial.suggest_float('rho', 0.9, 0.99),
            eps=trial.suggest_loguniform('eps', 1e-9, 1e-7),
            weight_decay=trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
        )
    elif optimizer_name == 'Nadam':
        optimizer = torch.optim.NAdam(
            model.parameters(),
            lr=learning_rate,
            betas=(trial.suggest_float('beta1', 0.85, 0.95), trial.suggest_float('beta2', 0.9, 0.999)),
            eps=trial.suggest_loguniform('eps', 1e-9, 1e-7),
            weight_decay=trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
        )
    elif optimizer_name == 'Radam':
        optimizer = optim.RAdam(
            model.parameters(),
            lr=learning_rate,
            betas=(trial.suggest_float('beta1', 0.85, 0.95), trial.suggest_float('beta2', 0.9, 0.999)),
            eps=trial.suggest_loguniform('eps', 1e-9, 1e-7),
            weight_decay=trial.suggest_loguniform('weight_decay', 1e-5, 1e-2)
        )

    loss = do_training(
        data_dir=args.data_dir,
        model_dir=args.model_dir,
        device=args.device,
        image_size=args.image_size,
        input_size=args.input_size,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        learning_rate=learning_rate,
        max_epoch=args.max_epoch,
        save_interval=args.save_interval,
        optimizer=optimizer,
        trial=trial  
    )
    return loss

def main(args):
    
    sampler = optuna.samplers.TPESampler()
    pruner = optuna.pruners.MedianPruner(n_startup_trials=n_startup_trials, 
                                         n_warmup_steps=n_warmup_steps, 
                                         interval_steps=interval_steps)
    optimizers = optims

    for optimizer_name in optimizers:
        print(f"Starting trials for optimizer: {optimizer_name}")
        study = optuna.create_study(direction='minimize', sampler=sampler, pruner=pruner)
        study.optimize(lambda trial: objective(trial, optimizer_name), n_trials=n_trials)


        print(f"Best trial for optimizer {optimizer_name}:")
        trial = study.best_trial
        print(f"  Value: {trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    
    
 
if __name__ == '__main__':
    args = parse_args()
    main(args)
