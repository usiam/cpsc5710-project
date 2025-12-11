import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd

class Cub2011Dataset(Dataset):
    """
    Dataset class for CUB-200-2011.
    It reads the official annotation files to load images, labels,
    and (optionally) bounding boxes.
    """
    def __init__(self, root_dir, transform=None, train=True, use_bbox=True, return_bbox=False): # <--- NEW use_bbox flag
        """
        Args:
            root_dir (string): Directory with all the CUB_200_2011 data.
            transform (callable, optional): Optional transform to be applied
                                            on a sample.
            train (bool): If True, creates training set, else creates test set.
            use_bbox (bool): If True, crops the image to the bounding box
                             before applying other transforms.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.use_bbox = use_bbox
        self.ret_bbox = return_bbox

        # Reads 'images.txt' (file_id, file_path)
        images_df = pd.read_csv(
            os.path.join(self.root_dir, 'images.txt'),
            sep=' ', names=['img_id', 'img_path']
        )

        # Reads 'image_class_labels.txt' (file_id, class_id)
        labels_df = pd.read_csv(
            os.path.join(self.root_dir, 'image_class_labels.txt'),
            sep=' ', names=['img_id', 'class_id']
        )
        
        # Reads 'train_test_split.txt' (file_id, is_train)
        split_df = pd.read_csv(
            os.path.join(self.root_dir, 'train_test_split.txt'),
            sep=' ', names=['img_id', 'is_train']
        )
        
        # Reads 'classes.txt' (class_id, class_name)
        classes_df = pd.read_csv(
            os.path.join(self.root_dir, 'classes.txt'),
            sep=' ', names=['class_id', 'class_name']
        )
        
        # --- NEW: Read bounding_boxes.txt ---
        bboxes_df = pd.read_csv(
            os.path.join(self.root_dir, 'bounding_boxes.txt'),
            sep=' ', names=['img_id', 'x', 'y', 'width', 'height']
        )

        # Create a mapping from class_id to class_name
        self.class_names = {row.class_id - 1: row.class_name.split('.')[-1] for row in classes_df.itertuples()}
        
        # Merge dataframes to link images, labels, and splits
        data_df = images_df.merge(labels_df, on='img_id')
        data_df = data_df.merge(split_df, on='img_id')
        data_df = data_df.merge(bboxes_df, on='img_id')

        # Filter for train or test
        if self.train:
            self.data = data_df[data_df['is_train'] == 1]
        else:
            self.data = data_df[data_df['is_train'] == 0]

        # self.samples will be a list of tuples: (full_image_path, class_label_index, bbox)
        self.samples = []
        for row in self.data.itertuples():
            img_path = os.path.join(self.root_dir, 'images', row.img_path)
            
            # The class_ids in the file are 1-indexed, but PyTorch expects 0-indexed labels
            class_label_index = row.class_id - 1 
            
            # Store the bounding box as a tuple (x, y, width, height)
            bbox = (row.x, row.y, row.width, row.height)
            
            self.samples.append((img_path, class_label_index, bbox))

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Fetches the sample at the given index.
        Returns:
            tuple: (image, label) where image is the transformed image
                   and label is the 0-indexed integer class label.
        """
        img_path, label, bbox = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.use_bbox:
            left = int(bbox[0])
            upper = int(bbox[1])
            right = int(bbox[0] + bbox[2])
            lower = int(bbox[1] + bbox[3])
            
            crop_box = (left, upper, right, lower)
            image = image.crop(crop_box)

        if self.transform:
            image = self.transform(image)
        
        if self.ret_bbox:
            return image, label, bbox
        else:
            return image, label



def create_dataloaders(
    root_dir,
    train_transform,
    test_transform,
    batch_size=32,
    use_bbox=True,
    ret_bbox=False,
    num_workers=10,
    train_loader_kwargs=None,
    test_loader_kwargs=None
):
    """
    Creates and returns training and testing DataLoaders for the CUB-200-2011 dataset.

    Args:
        root_dir (str): Path to the CUB_200_2011 dataset root directory.
        train_transform (callable): Transformations to apply to the training set.
        test_transform (callable): Transformations to apply to the test set.
        batch_size (int, optional): Number of samples per batch. Defaults to 32.
        use_bbox (bool, optional): Whether to crop images to their bounding boxes. Defaults to True.
        train_loader_kwargs (dict, optional): Additional keyword arguments for the training DataLoader
                                              (e.g., num_workers, pin_memory). Defaults to None.
        test_loader_kwargs (dict, optional): Additional keyword arguments for the test DataLoader.
                                             Defaults to None.

    Returns:
        tuple: (train_loader, test_loader)
    """

    if train_loader_kwargs is None:
        train_loader_kwargs = {}
    if test_loader_kwargs is None:
        test_loader_kwargs = {}

    train_dataset = Cub2011Dataset(
        root_dir=root_dir,
        transform=train_transform,
        train=True,
        use_bbox=use_bbox,
        return_bbox=ret_bbox
    )

    test_dataset = Cub2011Dataset(
        root_dir=root_dir,
        transform=test_transform,
        train=False,
        use_bbox=use_bbox,
        return_bbox=ret_bbox
    )

    train_shuffle = train_loader_kwargs.pop('shuffle', True)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=train_shuffle,
        num_workers=num_workers,
        **train_loader_kwargs
    )

    test_shuffle = test_loader_kwargs.pop('shuffle', False)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=test_shuffle,
        num_workers=num_workers,
        **test_loader_kwargs
    )

    return train_loader, test_loader