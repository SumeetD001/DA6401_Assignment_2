import torch
from PIL import Image
from torchvision import transforms

from models.multitask import MultiTaskPerceptionModel

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406],
                         std=[0.229,0.224,0.225])
])

def load_model():
    model = MultiTaskPerceptionModel().to(DEVICE)
    model.eval()
    return model

def predict(image_path):
    model = load_model()

    img = Image.open(image_path).convert("RGB")
    img = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        cls, box, seg = model(img)

    cls = torch.argmax(cls, dim=1).item()
    box = box.squeeze().cpu().numpy()
    seg = torch.sigmoid(seg).squeeze().cpu().numpy()

    return {
        'classification': cls_logits,
        'localization': bbox,
        'segmentation': seg_mask
    }


if __name__ == "__main__":
    image_path = "sample.jpg"
    cls, box, seg = predict(image_path)

    print("Class:", cls)
    print("BBox:", box)
    print("Seg shape:", seg.shape)
