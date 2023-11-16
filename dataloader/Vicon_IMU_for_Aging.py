import os
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

class VIFA(Dataset):
    def __init__(self, config, split='train'):
        self.root_dir = config['root_dir']
        self.data = pd.read_excel(os.path.join(self.root_dir, config['excel_file']))
        self.labels = []
        with open(os.path.join(self.root_dir, config['label_file']), 'r') as f:
            for line in f:
                self.labels.append(line.strip().split(' '))
        self.transforms = config['transforms']

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        rgb_path = os.path.join(self.root_dir, self.data.iloc[idx]['rgb_path'])
        rgb_image = Image.open(rgb_path).convert('RGB')
        excel_data = self.data.iloc[idx][['col1', 'col2', 'col3', 'col4', 'col5']].values.astype('float32')
        label = torch.tensor([int(x) for x in self.labels[idx]])

        if self.transforms:
            rgb_image = self.transforms(rgb_image)

        return {'rgb': rgb_image, 'excel': excel_data, 'label': label}
    
    @staticmethod
    def _collate_fn(batch):
        batch_size = len(batch)
        stu_xyzrs, stu_feas, stu_labels = zip(*[i['student'] for i in batch])
        tea_xyzrs, tea_feas, tea_labels = zip(*[i['teacher'] for i in batch])
        return {
            'student': (stu_xyzrs, stu_feas, stu_labels),
            'teacher': (tea_xyzrs, tea_feas, tea_labels),
        }


if __name__ == '__main':
    import yaml
    config_path = 'config/train/Normal_train.yaml.yaml'
    dataset_config_path = 'config/dataset/VIFA.yaml'
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    with open(dataset_config_path, 'r') as f:
        config['dataset'].update(yaml.safe_load(f))
    with open(dataset_config_path, 'r') as f:
        config['val_dataset'].update(yaml.safe_load(f))
    from torch.utils.data import DataLoader
    dataset = VIFA(split='train', config=config['dataset'])
    val_dataset = VIFA(split='valid', config=config['val_dataset'])

    loader = DataLoader(dataset=dataset, collate_fn=dataset._collate_fn, **config['train_dataloader'])
    val_loader = DataLoader(dataset=dataset, collate_fn=dataset._collate_fn, **config['val_dataloader'])
