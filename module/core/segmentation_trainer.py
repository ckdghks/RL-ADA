import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from module.core import get_model  # ✅ 사용자 정의 모델 로딩
import os
import copy

# from torchvision.utils import save_image
from PIL import Image
from ..utils.color_map import decode_segmap  # 또는 직접 정의
import numpy as np

def PretrainedSegModel(model_name='deeplabv3plus_resnet101', num_classes=19, output_stride=16, separable_conv=False, gpu_id=0, checkpoint_path=None):
    model = get_model(
        model=model_name,
        num_classes=num_classes,
        output_stride=output_stride,
        separable_conv=separable_conv
    )

    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    if checkpoint_path is not None:
        print(f"📦 Loading pretrained checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)

    return model

def freeze_bn(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()
            m.weight.requires_grad = False
            m.bias.requires_grad = False

def fine_tune_seg_model(model, dataloader, epochs=5, lr=1e-4, gpu_id=0):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model = copy.deepcopy(model)  # ✅ pretrained 모델을 보존
    model.train()
    freeze_bn(model)
    model.to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch in dataloader:
            images, labels = batch['images'].to(device), batch['labels'].to(device)

            unique_vals = torch.unique(labels)
            valid_labels = unique_vals[unique_vals != 255]
            if len(valid_labels) > 0 and valid_labels.max() >= model.classifier.final.out_channels:
                raise ValueError(f"❌ 라벨 값 {valid_labels.max().item()} 가 num_classes={model.classifier.final.out_channels} 보다 큽니다! {valid_labels.tolist()}")

            optimizer.zero_grad()
            outputs = model(images)['out'] if isinstance(model(images), dict) else model(images)
            loss = criterion(outputs, labels)
            # print(f"[Epoch {epoch}] Loss: {loss.item():.4f}")  # ✅ 로깅
            loss.backward()
            optimizer.step()

    return model

def evaluate_miou(model, dataset, num_classes, ignore_label=255, class_names=None, gpu_id=0, save_vis=False, vis_dir="eval_vis"):
    device = torch.device(f"cuda:{gpu_id}" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    hist = np.zeros((num_classes, num_classes), dtype=np.float64)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    os.makedirs(vis_dir, exist_ok=True)

    with torch.no_grad():
        for idx, sample in enumerate(dataloader):
            image = sample['images'].to(device)
            label = sample['labels'].to(device)

            output = model(image)
            output = output['out'] if isinstance(output, dict) else output
            pred_full = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
            gt_full = label.squeeze(0).cpu().numpy()

            mask = gt_full != ignore_label
            pred = pred_full[mask]
            gt = gt_full[mask]

            for p, g in zip(pred.flatten(), gt.flatten()):
                if g < num_classes:
                    hist[g, p] += 1

            if save_vis and idx < 10:
                pred_np = pred_full.astype(np.uint8)
                color_mask = decode_segmap(pred_np)
                img = Image.fromarray(color_mask)
                img.save(os.path.join(vis_dir, f"pred_{idx}.png"))

    iou_per_class = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-6)
    miou = np.mean(iou_per_class)

    print("Class-wise IoU:")
    for i, iou in enumerate(iou_per_class):
        class_label = class_names[i] if class_names and i < len(class_names) else f"Class {i}"
        print(f"  {class_label}: {iou:.4f}")

    print(f"\nMean IoU: {miou:.4f}")

    return miou, iou_per_class
