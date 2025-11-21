# train_compare.py
import argparse, time, torch, torchvision, requests, io
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import v2 as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor, FasterRCNN
from torchvision.ops import MultiScaleRoIAlign
from torchvision.models.detection.rpn import AnchorGenerator

from utils_pennfudan import PennFudanDataset, collate_fn

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_transforms(train):
    tfms = []
    if train:
        tfms.append(T.RandomHorizontalFlip(0.5))
    tfms.append(T.ToDtype(torch.float32, scale=True))
    return T.Compose(tfms)

# ---- Option 1: traditional Faster R-CNN with ResNet50-FPN backbone
def get_model_option1(num_classes=2):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    in_feat = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_feat, num_classes)
    return model
# ---- Option 2: Faster R-CNN with MobileNetV2 backbone
def get_model_option2(num_classes=2):
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    backbone.out_channels = 1280  
    anchor_generator = AnchorGenerator(
        sizes=((32, 64, 128, 256, 512),),  
        aspect_ratios=((0.5, 1.0, 2.0),)
    )
    roi_pooler = MultiScaleRoIAlign(featmap_names=["0"], output_size=7, sampling_ratio=2)
    model = FasterRCNN(
        backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler
    )
    return model

@torch.no_grad()
def evaluate_training_set(model, data_loader):
    model.eval()
    
    total = 0
    confs = []
    for images, targets in data_loader:
        images = [img.to(DEVICE) for img in images]
        outputs = model(images)
        for out, tgt in zip(outputs, targets):
            if len(out["scores"]) > 0:
                confs.append(out["scores"].mean().item())
            total += 1
    avg_conf = sum(confs) / max(len(confs), 1)
    return {"avg_score": avg_conf, "num_images": total}

def train_one_epoch(model, optimizer, data_loader, epoch, print_freq=20):
    model.train()
    lr_scheduler = None
    if epoch == 0:
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    running = 0.0
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(DEVICE) for img in images]
        targets = [{k: v.to(DEVICE) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        running += losses.item()
        if (i + 1) % print_freq == 0:
            print(f"[epoch {epoch}] iter {i+1}/{len(data_loader)} loss: {running/print_freq:.4f}")
            running = 0.0

    if lr_scheduler is not None:
        lr_scheduler.step()

from PIL import Image

def load_beatles_image(local_path):
    img = Image.open(local_path).convert("RGB")
    return img


@torch.no_grad()
def run_on_image(model, pil_img, score_thresh=0.5):
    model.eval()
    tfm = T.Compose([T.ToImage(), T.ToDtype(torch.float32, scale=True)])
    img = tfm(pil_img)
    out = model([img.to(DEVICE)])[0]
    keep = out["scores"] >= score_thresh
    boxes = out["boxes"][keep].cpu()
    labels = out["labels"][keep].cpu()
    scores = out["scores"][keep].cpu()
    return boxes, labels, scores

def draw_boxes(pil_img, boxes, labels, scores):
    draw = ImageDraw.Draw(pil_img)
    for b, l, s in zip(boxes, labels, scores):
        x1, y1, x2, y2 = b.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=(0,255,0), width=3)
        draw.text((x1, y1-10), f"{int(l.item())}:{s:.2f}", fill=(0,255,0))
    return pil_img

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/PennFudanPed", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=2, type=int)
    parser.add_argument("--lr", default=0.005, type=float)
    parser.add_argument("--weight_decay", default=0.0005, type=float)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument("--score_thresh", default=0.5, type=float)
    parser.add_argument("--beatles_path", type=str, required=True,
                    help="Path to local Beatles Abbey Road image file")

    args = parser.parse_args()

    train_ds = PennFudanDataset(args.data_root, transforms=get_transforms(train=True))
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn
    )

    # ---------- train Option 1 ----------
    model1 = get_model_option1(num_classes=2).to(DEVICE)
    optim1 = torch.optim.SGD(model1.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    print("\n===> Training Option 1 (Faster R-CNN ResNet50-FPN)")
    for epoch in range(args.epochs):
        train_one_epoch(model1, optim1, train_loader, epoch)

    train_stats1 = evaluate_training_set(model1, train_loader)
    print(f"[Option1] train-set avg_score@{args.score_thresh}: {train_stats1}")

    # ---------- train Option 2 ----------
    model2 = get_model_option2(num_classes=2).to(DEVICE)
    optim2 = torch.optim.SGD(model2.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)

    print("\n===> Training Option 2 (Faster R-CNN + MobileNetV2 backbone)")
    for epoch in range(args.epochs):
        train_one_epoch(model2, optim2, train_loader, epoch)

    train_stats2 = evaluate_training_set(model2, train_loader)
    print(f"[Option2] train-set avg_score@{args.score_thresh}: {train_stats2}")

    # ---------- (c) Beatles test ----------
    #beatles = load_beatles_image()  
    beatles = load_beatles_image(args.beatles_path)

    
    b1, l1, s1 = run_on_image(model1, beatles.copy(), score_thresh=args.score_thresh)
    b2, l2, s2 = run_on_image(model2, beatles.copy(), score_thresh=args.score_thresh)

    vis1 = draw_boxes(beatles.copy(), b1, l1, s1)
    vis2 = draw_boxes(beatles.copy(), b2, l2, s2)
    vis1.save("beatles_option1.jpg")
    vis2.save("beatles_option2.jpg")
    print("Saved: beatles_option1.jpg, beatles_option2.jpg")

   
    print(f"[Beatles] Option1: {len(b1)} boxes, mean score={float(s1.mean()) if len(s1)>0 else 0:.3f}")
    print(f"[Beatles] Option2: {len(b2)} boxes, mean score={float(s2.mean()) if len(s2)>0 else 0:.3f}")

if __name__ == "__main__":
    main()

