import os
import os.path as osp
import time
import math
from datetime import timedelta
from argparse import ArgumentParser
import requests

import torch
from torch import cuda
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from tqdm import tqdm

from east_dataset import EASTDataset
from dataset import SceneTextDataset
from model import EAST

# Import MLflow
import mlflow
import mlflow.pytorch  # For logging PyTorch models

def send_slack_notification(message):
    slack_webhook_url = "https://hooks.slack.com/services/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"  # Slack Webhook URL
    payload = {"text": message}
    requests.post(slack_webhook_url, json=payload)

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
    parser.add_argument('--max_epoch', type=int, default=150)
    parser.add_argument('--save_interval', type=int, default=5)
    args = parser.parse_args()

    if args.input_size % 32 != 0:
        raise ValueError('`input_size` must be a multiple of 32')

    return args

def do_training(data_dir, model_dir, device, image_size, input_size, num_workers, batch_size,
                learning_rate, max_epoch, save_interval):
    send_slack_notification("서버 1 모델 학습이 시작되었습니다.")  # 학습 시작 알림
    
    # Set up MLflow tracking URI and experiment name
    mlflow.set_tracking_uri("https://shoe-worker-recommends-boutique.trycloudflare.com")
    mlflow.set_experiment("EAST Model Training")

    # Log parameters with MLflow
    with mlflow.start_run(run_name="~~실험") as run:
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

        dataset = SceneTextDataset(
            data_dir,
            split='train',
            image_size=image_size,
            crop_size=input_size,
        )
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
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[max_epoch // 2], gamma=0.1)

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
                    
                    # Log batch-level metrics
                    mlflow.log_metrics({
                        "Cls loss": extra_info['cls_loss'],
                        "Angle loss": extra_info['angle_loss'],
                        "IoU loss": extra_info['iou_loss']
                    }, step=epoch * num_batches + pbar.n)

            # Log epoch-level metrics
            avg_loss = epoch_loss / num_batches
            mlflow.log_metric("Mean loss", avg_loss, step=epoch)

            scheduler.step()

            # 매 10 에폭마다 Slack 알림 전송
            if (epoch + 1) % 10 == 0:
                send_slack_notification(f"{epoch + 1} 에폭 완료: 평균 손실 {avg_loss:.4f}")

            print('Mean loss: {:.4f} | Elapsed time: {}'.format(
                avg_loss, timedelta(seconds=time.time() - epoch_start)))

            if (epoch + 1) % save_interval == 0:
                if not osp.exists(model_dir):
                    os.makedirs(model_dir)

                ckpt_fpath = osp.join(model_dir, 'latest.pth')
                torch.save(model.state_dict(), ckpt_fpath)
                
                # Log model checkpoint as an artifact
                mlflow.log_artifact(ckpt_fpath)

        # 학습 완료 후 Slack 알림 전송
        send_slack_notification("서버 1 모델 학습이 완료되었습니다!")
        
        # Log final model
        mlflow.pytorch.log_model(model, "model")

def main(args):
    do_training(**args.__dict__)

if __name__ == '__main__':
    args = parse_args()
    main(args)