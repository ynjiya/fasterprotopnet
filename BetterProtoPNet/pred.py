import argparse
from pathlib import Path

import torch
from torchvision import transforms
from PIL import Image
from settings import *


def parse_index_file(index_file: Path) -> dict:
    out = {}
    with open(index_file) as f:
        for line in f.readlines():
            index, path = line.split()
            index = int(index)
            path = Path(path)
            out[index] = str(path)
    return out


def main(image_path: str, model_path: str):
    index_dict = parse_index_file(Path('./datasets/CUB_200_2011/classes.txt'))

    # Load model
    model = torch.load(model_path)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    image = Image.open(image_path).convert('RGB')

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    normalize = transforms.Normalize(mean=mean,
                                     std=std)
    _transforms = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        normalize,
    ])
    image_tensor = _transforms(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, min_distances = model(image_tensor)

    _, predicted_class = torch.max(logits, 1)

    print(index_dict[predicted_class.item() + 1][4:])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predict the class of a bird image.')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('model_path', type=str, help='Path to the model file')
    args = parser.parse_args()

    main(args.image_path, args.model_path)
