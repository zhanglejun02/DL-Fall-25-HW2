# utils_pennfudan.py
import os, numpy as np, torch, PIL
from PIL import Image
from torchvision.tv_tensors import BoundingBoxes, Mask, Image as TvImage
from torchvision.ops import box_convert

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms=None):
        self.root = root
        self.transforms = transforms
        self.imgs = list(sorted(os.listdir(os.path.join(root, "PNGImages"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "PedMasks"))))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])

        img = PIL.Image.open(img_path).convert("RGB")
        mask = PIL.Image.open(mask_path)  
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids != 0]   
        masks = (mask == obj_ids[:, None, None]).astype(np.uint8)
        boxes = []
        for m in masks:
            pos = np.where(m)
            xmin = np.min(pos[1]); xmax = np.max(pos[1])
            ymin = np.min(pos[0]); ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((len(obj_ids),), dtype=torch.int64)  #person
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(obj_ids),), dtype=torch.int64)

        target = {
            "boxes": BoundingBoxes(boxes, format="XYXY", canvas_size=(img.height, img.width)),
            "labels": labels,
            "masks": Mask(masks),
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        img = TvImage(torch.ByteTensor(torch.ByteStorage.from_buffer(img.tobytes())).view(img.height, img.width, 3).permute(2,0,1))
        

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

def collate_fn(batch):
    return tuple(zip(*batch))
    
