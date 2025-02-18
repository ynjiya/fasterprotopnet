import argparse
from pathlib import Path
import shutil
import sys
import os


def read_train_test_split(train_test_split_file: Path):
    split_info = {}
    with open(train_test_split_file, "r") as file:
        for line in file:
            image_id, is_training_image = line.strip().split()
            split_info[int(image_id)] = int(is_training_image)
    return split_info


def read_images(file_path: Path):
    images = {}
    with open(file_path, "r") as file:
        for line in file:
            image_id, image_path = line.strip().split(maxsplit=1)
            images[int(image_id)] = image_path
    return images


def get_train_test_split(split_info: dict, images: dict):
    train_set = []
    test_set = []
    for image_id, is_training_image in split_info.items():
        if is_training_image == 1:
            train_set.append(images[image_id])
        else:
            test_set.append(images[image_id])
    return train_set, test_set


def split_datasets(
    index_file: Path, train_test_split_file: Path, images_path: Path, dest_path: Path
):
    split_info = read_train_test_split(train_test_split_file)
    images = read_images(index_file)
    train_set, test_set = get_train_test_split(split_info, images)

    TRAIN_CROPPED_DIR = dest_path / "train_cropped"
    TEST_CROPPED_DIR = dest_path / "test_cropped"

    TRAIN_CROPPED_DIR.mkdir(parents=True, exist_ok=True)
    TEST_CROPPED_DIR.mkdir(parents=True, exist_ok=True)

    for image_path in train_set:
        tmp = image_path
        image_path = images_path / image_path
        dest_image_path = TRAIN_CROPPED_DIR / tmp
        if not os.path.exists(dest_image_path.parent):
            os.mkdir(dest_image_path.parent)
        shutil.copy(image_path, dest_image_path)

    for image_path in test_set:
        tmp = image_path
        image_path = images_path / image_path
        dest_image_path = TEST_CROPPED_DIR / tmp
        if not os.path.exists(dest_image_path.parent):
            os.mkdir(dest_image_path.parent)
        shutil.copy(image_path, dest_image_path)


def main():
    parser = argparse.ArgumentParser(
        description="Split dataset into training and test sets"
    )
    parser.add_argument("index_file", type=Path, help="Path to the index file")
    parser.add_argument(
        "train_test_split_file", type=Path, help="Path to the train/test split file"
    )
    parser.add_argument("images_path", type=Path, help="Path to the images directory")
    parser.add_argument(
        "dest_path",
        type=Path,
        help="Path to the destination directory",
        default=Path("datasets/cub200_cropped"),
    )
    args = parser.parse_args(sys.argv[1:])

    split_datasets(
        args.index_file, args.train_test_split_file, args.images_path, args.dest_path
    )


if __name__ == "__main__":
    main()
