import sys
from pathlib import Path
import argparse
from PIL import Image


def crop_image(image_path: Path, coords: tuple, saved_location: Path):
    """
    Crop the image and save it to the target location
    :param image_path: the path to the image
    :param coords: a tuple of x1, y1, x2, y2
    :param saved_location: the path to save the cropped image
    """
    with Image.open(image_path) as img:
        dpi = img.info.get('dpi', (72, 72))
    image_obj = Image.open(image_path)
    cropped_image = image_obj.crop(coords)
    if not saved_location.parent.exists():
        saved_location.parent.mkdir(parents=True)
    cropped_image.save(saved_location, dpi=dpi)


def parse_index_file(index_file: Path) -> dict:
    out = {}
    with open(index_file) as f:
        for line in f.readlines():
            index, path = line.split()
            index = int(index)
            path = Path(path)
            out[index] = path
    return out


def parse_coordinate_file(coordinate_file: Path) -> dict:
    out = {}
    with open(coordinate_file) as f:
        for line in f.readlines():
            index, *coords = line.split()
            index = int(index)
            coords = tuple(map(float, coords))
            coords = (coords[0], coords[1], coords[0] + coords[2], coords[1] + coords[3])
            out[index] = coords
    return out


def crop_dataset(dataset_path: Path, index_file: Path, coordinate_file: Path, dest_path: Path):
    indexes = parse_index_file(index_file)
    coords = parse_coordinate_file(coordinate_file)
    for index, path in indexes.items():
        try:
            crop_image(dataset_path / path, coords[index], dest_path / path)
        except ValueError as e:
            print(f'Error cropping image {path}: {e}', file=sys.stderr)
            raise

def main():
    parser = argparse.ArgumentParser(description='Crop the dataset')
    parser.add_argument('dataset_path', type=Path, help='the path to the dataset')
    parser.add_argument('index_file', type=Path, help='the path to the index file')
    parser.add_argument('coordinate_file', type=Path, help='the path to the coordinate file')
    parser.add_argument('dest_path', type=Path, help='the path to save the cropped dataset')
    args = parser.parse_args(sys.argv[1:])

    crop_dataset(args.dataset_path, args.index_file, args.coordinate_file, args.dest_path)


if __name__ == '__main__':
    main()
