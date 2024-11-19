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

import mlflow
import mlflow.pytorch

from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import random

'''
    server_number : 사용중인 서버숫자
    name : 작업자 이름
    status : 학습중, 학습완료
    task : 어떤 작업을 수행하는가
    mlflow url : mlflow 주소
    mlflow experiment : mlflow 실험이름
    mlflow_runname : mlflow 런이름
    uuid_path : 카카오톡 uuid가 저장된 json 경로
'''

server_number = 1
name = ""
task = ""
mlflow_url = ""
mlflow_exp = ""
mlflow_runname = ""
uuid_path = ""

with open(uuid_path, 'r') as fp:
    receiver_uuids = json.load(fp)

receiver_uuids = list(receiver_uuids.keys())

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('--data_dir', type=str,
                        default=os.environ.get('SM_CHANNEL_TRAIN', 'data'))
    parser.add_argument('--model_dir', type=str, default=os.environ.get('SM_MODEL_DIR',
                                                                        'trained_models'))

    parser.add_argument('--device', default='cuda' if cuda.is_available() else 'cpu')
    parser.add_argument('--num_workers', type=int, default=8)

    parser.add_argument('--image_size', type=int, default=2048)
    parser.add_argument('--input_size', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--visualize', action='store_true', help="Enable visualization of augmented images")
    
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval, visualize=False):
    
    sp.update_server_status(server_number = server_number, 
                            name = name,
                            status = True, 
                            task=task)
    
    kakao.send_message(receiver_uuids=receiver_uuids, 
                       message_text=f"{name}님이\n서버 {server_number}번에서\n{max_epoch}epoch\n{task}학습을 시작하였습니다.")
    
    slack.send_slack_notification(f"{name}님이\n서버 {server_number}번에서\n{max_epoch}epoch\n{task}학습을 시작하였습니다.")
    mlflow.set_tracking_uri(mlflow_url)
    mlflow.set_experiment(mlflow_exp)
    
    epoch_loss = 0 
    num_batches = 1 
    best_loss = float('inf')
    
    with mlflow.start_run(run_name = mlflow_runname) as run:
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
        try:
            dataset = SceneTextDataset(
                data_dir,
                split='train',
                image_size=image_size,
                crop_size=input_size,
            )
            
            if visualize:
                sample_indices = random.sample(range(len(dataset)), 8)
                fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
                for i, idx in enumerate(sample_indices):
                    img, word_bboxes, _ = dataset[idx]
                    if img.shape[0] == 3:
                        img = img.transpose(1, 2, 0)
                    img = np.clip((img * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
    
                    axes[i // 4, i % 4].imshow(img)
                    for bbox in word_bboxes:
                        if bbox.shape == (4, 2):
                            polygon = plt.Polygon(bbox, closed=True, edgecolor='red', linewidth=2, fill=None)
                            axes[i // 4, i % 4].add_patch(polygon)
    
                    axes[i // 4, i % 4].axis('off')
                    axes[i // 4, i % 4].set_title(f"Sample Image {i + 1}")
                    Image.fromarray(img).save(f"sample_image_{i + 1}.jpg")
    
                plt.tight_layout()
                plt.show()
            
            dataset = EASTDataset(dataset)
            num_batches = math.ceil(len(dataset) / batch_size)
            train_loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers
            )

            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = EAST()
            model.to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.0001)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

            model.train()
            for epoch in range(max_epoch):
                epoch_loss, epoch_start = 0, time.time()
                with tqdm(total=num_batches) as pbar:
                    for img, gt_score_map, gt_geo_map, roi_mask in train_loader:
                        pbar.set_description('[Epoch {}]'.format(epoch + 1))

                        loss, extra_info = model.train_step(img, gt_score_map, gt_geo_map, roi_mask)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                        loss_val = loss.item()
                        epoch_loss += loss_val

                        pbar.update(1)
                        val_dict = {
                            'Cls loss': extra_info['cls_loss'], 'Angle loss': extra_info['angle_loss'],
                            'IoU loss': extra_info['iou_loss']
                        }
                        pbar.set_postfix(val_dict)
                        
                        mlflow.log_metrics({
                        "Cls loss": extra_info['cls_loss'],
                        "Angle loss": extra_info['angle_loss'],
                        "IoU loss": extra_info['iou_loss']
                        }, step=epoch * num_batches + pbar.n)
                        
                avg_loss = epoch_loss / num_batches
                mlflow.log_metric("Mean loss", avg_loss, step=epoch)
                
                if avg_loss < best_loss:
                    best_loss = avg_loss
                    best_ckpt_path = osp.join(model_dir, 'best.pth')
                    torch.save(model.state_dict(), best_ckpt_path)
                    mlflow.log_artifact(best_ckpt_path)
                
                scheduler.step()

                print('Mean loss: {:.4f} | Elapsed time: {}'.format(
                    epoch_loss / num_batches, timedelta(seconds=time.time() - epoch_start)))

                if (epoch + 1) % save_interval == 0:
                    if not osp.exists(model_dir):
                        os.makedirs(model_dir)

                    ckpt_fpath = osp.join(model_dir, 'best.pth')
                    torch.save(model.state_dict(), ckpt_fpath)
                    
                    mlflow.log_artifact(ckpt_fpath)
                    
                if (epoch + 1) % (max_epoch // 3) == 0:
                    kakao.send_message(receiver_uuids=receiver_uuids,
                                    message_text=f"서버 {server_number}번\n{name}님의 {task}\n학습 현황 Epoch: {epoch + 1}\nloss: {best_loss}")
                
            train_end_time = time.time()
            elapsed_time = train_end_time - train_start_time    
            formatted_elapsed_time = str(timedelta(seconds=int(elapsed_time)))
            kakao.send_message(receiver_uuids=receiver_uuids,
                        message_text=f"서버 {server_number} 학습 완료.\n{name}님의 {task} 학습 결과\nEpoch: {max_epoch}\nBest Loss: {best_loss}\n경과 시간 : {formatted_elapsed_time}")
            slack.send_slack_notification(f"{name}님이\n서버 {server_number}번에서\n{max_epoch}epoch\n{task} 학습이 완료되었습니다.")
            mlflow.pytorch.log_model(model, "model")
            
        except Exception as e:
            kakao.send_message(receiver_uuids=receiver_uuids,
                            message_text=f"서버 {server_number}번에서\n{name}님의 {task} 학습 도중\n에러가 발생하였습니다.\n확인이 필요합니다.\n{e}")
            slack.send_slack_notification(f"서버 {server_number}번에서\n{name}님의 {task} 학습 도중\n에러가 발생하였습니다.\n{e}")
        finally:
            data = {
                "epoch" : max_epoch,
                "loss" : f"{best_loss}",
                "task" : task
            }
            
            sp.append_training_log(name, data)
            sp.update_server_status(server_number = server_number,
                                    status = False)
    
def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)
