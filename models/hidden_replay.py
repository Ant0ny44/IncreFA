import logging
from typing import Dict, List, Optional, Tuple

import torch


def herding_select(
    features: torch.Tensor, labels: torch.Tensor, budget_per_class: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    selected_features = []
    selected_labels = []
    for class_id in labels.unique():
        class_features = features[labels == class_id]
        count = min(budget_per_class, class_features.size(0))
        if count == class_features.size(0):
            selected_features.append(class_features)
            selected_labels.append(torch.full((count,), class_id, dtype=labels.dtype))
            continue
        target_mean = class_features.mean(dim=0)
        running_sum = torch.zeros_like(target_mean)
        chosen = []
        for step in range(1, count + 1):
            distances = ((running_sum + class_features - step * target_mean) ** 2).sum(dim=1)
            if chosen:
                distances[chosen] = float('inf')
            index = distances.argmin().item()
            chosen.append(index)
            running_sum = running_sum + class_features[index]
        indices = torch.tensor(chosen, dtype=torch.long)
        selected_features.append(class_features[indices])
        selected_labels.append(torch.full((count,), class_id, dtype=labels.dtype))
    return torch.cat(selected_features), torch.cat(selected_labels)


class ReplayBuffer:
    def __init__(self, buffer_size_per_task: int = 150):
        self.buffer_size_per_task = buffer_size_per_task
        self.buffer_size_per_class = buffer_size_per_task
        self.task_tensors: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}
        self.current_task_id = 0
        self._cache_all_x: Optional[torch.Tensor] = None
        self._cache_all_y: Optional[torch.Tensor] = None
        self._cache_excl_x: Optional[torch.Tensor] = None
        self._cache_excl_y: Optional[torch.Tensor] = None
        self._cache_dirty = True

    def _invalidate_cache(self):
        self._cache_all_x = None
        self._cache_all_y = None
        self._cache_excl_x = None
        self._cache_excl_y = None
        self._cache_dirty = True

    def _build_cache(self):
        if not self._cache_dirty:
            return
        all_x, all_y, old_x, old_y = [], [], [], []
        for task_id, (features, labels) in self.task_tensors.items():
            all_x.append(features)
            all_y.append(labels)
            if task_id != self.current_task_id:
                old_x.append(features)
                old_y.append(labels)
        self._cache_all_x = torch.cat(all_x) if all_x else None
        self._cache_all_y = torch.cat(all_y) if all_y else None
        self._cache_excl_x = torch.cat(old_x) if old_x else None
        self._cache_excl_y = torch.cat(old_y) if old_y else None
        self._cache_dirty = False

    def add_task_samples(self, task_id: int, samples: List[Tuple[torch.Tensor, torch.Tensor]]):
        if not samples:
            return
        features = torch.stack([feature for feature, _ in samples])
        labels = torch.stack([label for _, label in samples])
        limit = self.buffer_size_per_class * labels.unique().numel()
        if features.size(0) > limit:
            features, labels = herding_select(
                features, labels, self.buffer_size_per_class
            )
            logging.info("task %s replay exemplars: %s", task_id, features.size(0))
        self.task_tensors[task_id] = (features, labels)
        self._invalidate_cache()

    def sample_replay_batch(self, batch_size: int, exclude_current_task: bool = True):
        self._build_cache()
        features = self._cache_excl_x if exclude_current_task else self._cache_all_x
        labels = self._cache_excl_y if exclude_current_task else self._cache_all_y
        if features is None or not features.size(0):
            return None, None
        count = min(batch_size, features.size(0))
        indices = torch.randperm(features.size(0))[:count]
        return features[indices], labels[indices]

    def set_current_task(self, task_id: int):
        if task_id != self.current_task_id:
            self.current_task_id = task_id
            self._invalidate_cache()

    def get_total_samples(self) -> int:
        return sum(features.size(0) for features, _ in self.task_tensors.values())

    def get_task_sample_counts(self) -> Dict[int, int]:
        return {task_id: features.size(0) for task_id, (features, _) in self.task_tensors.items()}
