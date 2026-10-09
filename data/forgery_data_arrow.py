import logging

import numpy as np
import torch
from datasets import load_from_disk
from PIL import Image
from torch.utils.data import Dataset, Subset
from torchvision import transforms


class ForgeryArrowDataset(Dataset):
    def __init__(
        self,
        arrow_data_dir: str,
        real_label_name: str,
        task_sizes: list[int],
        time_series: list[str],
        holdout_labels: list[str],
        split_ratio: float,
        split_seed: int,
        split_counts: dict,
    ):
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.48145466, 0.4578275, 0.40821073],
                std=[0.26862954, 0.26130258, 0.27577711],
            ),
        ])
        self.dataset = load_from_disk(arrow_data_dir)
        self._build_label_mapping(real_label_name)

        missing = sorted(set(holdout_labels) - set(self.class_names))
        if missing:
            raise ValueError(f"Unknown holdout labels: {missing}")
        self.heldout_classes = [
            self.label_str_to_idx[name] for name in holdout_labels
        ]
        self.all_classes = [
            class_id for class_id in range(len(self.class_names))
            if class_id not in self.heldout_classes
        ]
        self.learned_classes = []

        labels = np.asarray(self.dataset['label'], dtype=object)
        self.original_label_indices = np.fromiter(
            (self.label_str_to_idx[label] for label in labels),
            dtype=np.int32,
            count=len(labels),
        )
        self.train_indices_by_class = {}
        self.test_indices_by_class = {}
        rng = np.random.default_rng(split_seed)
        for class_name, class_id in self.label_str_to_idx.items():
            indices = rng.permutation(
                np.flatnonzero(self.original_label_indices == class_id)
            )
            if class_name in split_counts:
                train_count = int(split_counts[class_name]['train'])
                test_count = int(split_counts[class_name]['test'])
                if train_count + test_count > len(indices):
                    raise ValueError(
                        f"Requested {train_count + test_count} samples for "
                        f"{class_name}, found {len(indices)}"
                    )
            else:
                train_count = int(len(indices) * split_ratio)
                test_count = len(indices) - train_count
            self.train_indices_by_class[class_id] = indices[:train_count]
            self.test_indices_by_class[class_id] = indices[
                train_count:train_count + test_count
            ]

        flat_order = [self.real_num] + [
            self.label_str_to_idx[name] for name in time_series
            if self.label_str_to_idx[name] not in self.heldout_classes
        ]
        if len(flat_order) != len(set(flat_order)):
            raise ValueError("Incremental class order contains duplicates")
        if set(flat_order) != set(self.all_classes):
            raise ValueError("Incremental class order does not match known classes")
        if sum(task_sizes) != len(flat_order) or any(size <= 0 for size in task_sizes):
            raise ValueError("task_sizes must partition all known classes")

        self.label_mapping = {
            original_id: incremental_id
            for incremental_id, original_id in enumerate(flat_order)
        }
        self.order = []
        offset = 0
        for size in task_sizes:
            self.order.append(flat_order[offset:offset + size])
            offset += size

        logging.info(
            "Loaded %d samples, %d known classes and %d held-out classes",
            len(self.dataset), len(self.all_classes), len(self.heldout_classes),
        )

    def _build_label_mapping(self, real_label_name: str) -> None:
        self.class_names = sorted(set(self.dataset['label']))
        self.label_str_to_idx = {
            name: class_id for class_id, name in enumerate(self.class_names)
        }
        self.idx_to_label_str = {
            class_id: name for name, class_id in self.label_str_to_idx.items()
        }
        if real_label_name not in self.label_str_to_idx:
            raise ValueError(f"Real label {real_label_name!r} not found")
        self.real_num = self.label_str_to_idx[real_label_name]

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        item = self.dataset[index]
        image = item['image']
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        image = self.transform(image.convert('RGB'))
        return image, self.label_str_to_idx[item['label']]

    def incremental_learning(self):
        for classes in self.order:
            yield self._generate_subset(classes)

    def _generate_subset(self, current_classes: list[int]):
        self.learned_classes.extend(current_classes)
        train_indices = np.concatenate([
            self.train_indices_by_class[class_id]
            for class_id in current_classes
        ])
        test_indices = np.concatenate([
            self.test_indices_by_class[class_id]
            for class_id in self.learned_classes
        ])
        unseen_indices = np.concatenate([
            self.test_indices_by_class[class_id]
            for class_id in self.heldout_classes
        ])
        metadata = {
            'current_classes': current_classes,
            'learned_classes': list(self.learned_classes),
            'unseen_classes': list(self.heldout_classes),
        }
        return (
            Subset(self, train_indices),
            Subset(self, test_indices),
            Subset(self, unseen_indices),
            metadata,
        )

    def apply_label_mapping_batch(self, labels: torch.Tensor) -> torch.Tensor:
        labels = torch.as_tensor(labels)
        return torch.tensor(
            [self.label_mapping[label.item()] for label in labels.flatten()],
            dtype=labels.dtype,
            device=labels.device,
        ).reshape(labels.shape)

    @property
    def mapped_real_label(self) -> int:
        return self.label_mapping[self.real_num]
