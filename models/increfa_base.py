"""Paper-aligned IncreFA base model without dynamic family discovery."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.clip.clip import load
from models.hidden_replay import ReplayBuffer


class PaperHierarchicalSpace(nn.Module):
    """Projection and learnable model/family anchors from Eqs. 5-9."""

    def __init__(self, input_dim: int, hidden_units: int, num_classes: int,
                 num_families: int, fine_weight: float, coarse_weight: float):
        super().__init__()
        self.num_classes = num_classes
        self.num_families = num_families
        self.fine_weight = fine_weight
        self.coarse_weight = coarse_weight

        # Figure 3 specifies a linear projection followed by LayerNorm.
        self.projector = nn.Sequential(
            nn.Linear(input_dim, hidden_units),
            nn.LayerNorm(hidden_units),
        )

        self.model_anchors = nn.Parameter(torch.empty(num_classes, hidden_units))
        self.family_anchors = nn.Parameter(torch.empty(num_families, hidden_units))
        nn.init.orthogonal_(self.model_anchors)
        nn.init.orthogonal_(self.family_anchors)

        self.register_buffer(
            'class_family_ids',
            torch.full((num_classes,), -1, dtype=torch.long),
        )
        self.learning_indicator = 0
        self.last_fine_loss = 0.0
        self.last_coarse_loss = 0.0

    def set_class_family_mapping(self, class_to_family: Dict[int, int]) -> None:
        for class_id, family_id in class_to_family.items():
            if not 0 <= class_id < self.num_classes:
                raise ValueError(f"Invalid class ID {class_id}")
            if not 0 <= family_id < self.num_families:
                raise ValueError(f"Invalid family ID {family_id}")
            self.class_family_ids[class_id] = family_id

    def update_indicator(self, new_indicator: int) -> None:
        self.learning_indicator = new_indicator

    @staticmethod
    def _orthogonality_loss(anchors: torch.Tensor) -> torch.Tensor:
        anchors = F.normalize(anchors, dim=-1)
        gram = anchors @ anchors.T
        identity = torch.eye(gram.size(0), device=gram.device, dtype=gram.dtype)
        return (gram - identity).pow(2).sum()

    def hierarchical_loss(self, z: torch.Tensor,
                          labels: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        unique_labels = labels.unique(sorted=True)
        prototypes = torch.stack([
            F.normalize(z[labels == class_id].mean(dim=0), dim=0)
            for class_id in unique_labels
        ])

        model_anchors = F.normalize(self.model_anchors[unique_labels], dim=-1)
        fine_alignment = (1.0 - (prototypes * model_anchors).sum(dim=-1)).sum()
        active_model_anchors = self.model_anchors[:self.learning_indicator]
        fine_loss = fine_alignment + self._orthogonality_loss(active_model_anchors)

        family_ids = self.class_family_ids[unique_labels]
        if (family_ids < 0).any():
            missing = unique_labels[family_ids < 0].tolist()
            raise RuntimeError(f"Missing predefined family IDs for classes {missing}")

        active_family_ids = self.class_family_ids[:self.learning_indicator].unique()
        active_family_ids = active_family_ids[active_family_ids >= 0]
        coarse_alignment_terms = []
        for family_id in family_ids.unique(sorted=True):
            family_proto = F.normalize(
                prototypes[family_ids == family_id].mean(dim=0), dim=0
            )
            family_anchor = F.normalize(self.family_anchors[family_id], dim=0)
            coarse_alignment_terms.append(1.0 - torch.dot(family_proto, family_anchor))
        coarse_alignment = torch.stack(coarse_alignment_terms).sum()
        coarse_loss = coarse_alignment + self._orthogonality_loss(
            self.family_anchors[active_family_ids]
        )
        return fine_loss, coarse_loss

    def forward(self, features: torch.Tensor,
                labels: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.projector(features)
        if labels is None:
            return z, z.new_zeros(())

        fine_loss, coarse_loss = self.hierarchical_loss(z, labels)
        self.last_fine_loss = fine_loss.detach().item()
        self.last_coarse_loss = coarse_loss.detach().item()
        weighted_loss = self.fine_weight * fine_loss + self.coarse_weight * coarse_loss
        return z, weighted_loss


class IncreFABaseClassifier(nn.Module):
    """IncreFA equations 5-15 with a fixed, predefined family taxonomy."""

    def __init__(self, backbone: str = 'ViT-L/14', feature_layer: str = 'layer11',
                 device: str = 'cuda:0', num_predictions: int = 27,
                 input_dim: int = 1024, hidden_units: int = 128,
                 num_families: int = 4, buffer_size_per_task: int = 150,
                 unseen_threshold: float = 0.65,
                 fine_loss_weight: float = 0.2, coarse_loss_weight: float = 0.5,
                 unseen_beta_min: float = 0.0, unseen_beta_max: float = 1.0):
        super().__init__()
        self.backbone, _ = load(backbone, device=device)
        self.feature_layer = feature_layer
        self.device = device
        self.num_families = num_families
        self.unseen_threshold = unseen_threshold
        self.unseen_beta_min = unseen_beta_min
        self.unseen_beta_max = unseen_beta_max
        self.class_to_family: Dict[int, int] = {}
        self.family_names: Dict[int, str] = {}

        self.orth_space = PaperHierarchicalSpace(
            input_dim=input_dim,
            hidden_units=hidden_units,
            num_classes=num_predictions,
            num_families=num_families,
            fine_weight=fine_loss_weight,
            coarse_weight=coarse_loss_weight,
        )
        self.fc = nn.Linear(hidden_units, num_predictions)
        nn.init.xavier_uniform_(self.fc.weight, gain=nn.init.calculate_gain('linear'))

        self.replay_buffer = ReplayBuffer(
            buffer_size_per_task=buffer_size_per_task,
        )
        self.current_task_samples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.current_task_by_class: Dict[int, List[torch.Tensor]] = {}

    def set_class_family_mapping(self, class_to_family: Dict[int, int],
                                 family_names: Dict[int, str]) -> None:
        self.class_to_family = dict(class_to_family)
        self.family_names = dict(family_names)
        self.orth_space.set_class_family_mapping(class_to_family)
        logging.info(
            "[IncreFABase] Loaded predefined taxonomy for %d classes: %s",
            len(class_to_family), family_names,
        )

    def _extract_features(self, batch: torch.Tensor) -> torch.Tensor:
        if batch.dim() != 4:
            return batch
        with torch.no_grad():
            _, layer_outputs = self.backbone.encode_image(batch)
            return layer_outputs[self.feature_layer].float()

    def forward(self, batch: torch.Tensor, y_gt: Optional[torch.Tensor],
                store_samples: bool = False
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self._extract_features(batch)
        z, hierarchy_loss = self.orth_space(features, y_gt)
        logits = self.fc(z)
        if store_samples and y_gt is not None:
            cpu_features = features.detach().cpu()
            cpu_labels = y_gt.detach().cpu()
            for feature, label in zip(cpu_features.unbind(0), cpu_labels.unbind(0)):
                self.current_task_samples.append((feature, label))
                self.current_task_by_class.setdefault(label.item(), []).append(feature)
        return logits, hierarchy_loss

    def replay_forward(self, replay_batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        features = self._extract_features(replay_batch)
        z = self.orth_space.projector(features)
        return self.fc(z), z.new_zeros(())

    def finalize_task(self, task_id: int) -> None:
        if self.current_task_samples:
            self.replay_buffer.add_task_samples(task_id, self.current_task_samples)
            self.current_task_samples = []
            self.current_task_by_class = {}
            self.replay_buffer.set_current_task(task_id)

    def get_replay_batch(self, batch_size: int
                         ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        features, labels = self.replay_buffer.sample_replay_batch(batch_size)
        if features is None:
            return None, None
        return features.to(self.device), labels.to(self.device)

    def get_buffer_status(self) -> Dict[str, object]:
        """Expose replay state through the trainer's shared logging API."""
        return {
            'total_samples': self.replay_buffer.get_total_samples(),
            'task_counts': self.replay_buffer.get_task_sample_counts(),
            'current_task': self.replay_buffer.current_task_id,
            'buffer_size_per_task': self.replay_buffer.buffer_size_per_task,
            'num_families': self.num_families,
        }

    def generate_unseen_samples(
        self,
        batch_size: int,
        num_mix: int = 2,
        current_mix_fraction: float = 0.0,
    ) -> Optional[torch.Tensor]:
        """Eq. 11 with optional current-vs-replay interpolation.

        The current task is not added to replay until evaluation finishes. A
        non-zero ``current_mix_fraction`` prevents the unseen objective from
        ignoring the newest classes during training.
        """
        if num_mix != 2:
            raise ValueError('Paper-aligned unseen mixing requires num_mix=2')
        if not 0.0 <= current_mix_fraction <= 1.0:
            raise ValueError('current_mix_fraction must be in [0, 1]')
        self.replay_buffer._build_cache()
        features = self.replay_buffer._cache_all_x
        labels = self.replay_buffer._cache_all_y
        if features is None or labels is None:
            return None

        replay_classes = labels.unique()
        current_classes = sorted(self.current_task_by_class)
        can_mix_current = bool(current_classes) and replay_classes.numel() > 0
        can_mix_replay = replay_classes.numel() >= 2
        if not can_mix_current and not can_mix_replay:
            return None

        current_count = int(round(batch_size * current_mix_fraction))
        if not can_mix_current:
            current_count = 0
        if not can_mix_replay:
            current_count = batch_size
        replay_count = batch_size - current_count

        first_features, second_features = [], []
        for _ in range(current_count):
            current_class = current_classes[
                torch.randint(len(current_classes), (1,)).item()
            ]
            current_pool = self.current_task_by_class[current_class]
            first_features.append(
                current_pool[torch.randint(len(current_pool), (1,)).item()]
            )
            replay_class = replay_classes[
                torch.randint(replay_classes.numel(), (1,)).item()
            ]
            replay_pool = torch.where(labels == replay_class)[0]
            second_features.append(
                features[replay_pool[torch.randint(replay_pool.numel(), (1,)).item()]]
            )

        if replay_count:
            first_pos = torch.randint(replay_classes.numel(), (replay_count,))
            second_pos = torch.randint(replay_classes.numel() - 1, (replay_count,))
            second_pos += (second_pos >= first_pos).long()
            for first_class, second_class in zip(
                replay_classes[first_pos], replay_classes[second_pos]
            ):
                first_pool = torch.where(labels == first_class)[0]
                second_pool = torch.where(labels == second_class)[0]
                first_features.append(
                    features[first_pool[torch.randint(first_pool.numel(), (1,)).item()]]
                )
                second_features.append(
                    features[second_pool[torch.randint(second_pool.numel(), (1,)).item()]]
                )

        first_features = torch.stack(first_features).to(self.device)
        second_features = torch.stack(second_features).to(self.device)
        z1 = self.orth_space.projector(first_features)
        z2 = self.orth_space.projector(second_features)
        beta = self.unseen_beta_min + (
            self.unseen_beta_max - self.unseen_beta_min
        ) * torch.rand(batch_size, 1, device=self.device)
        return beta * z1 + (1.0 - beta) * z2

    def compute_unseen_loss(self, unseen_z: torch.Tensor,
                            num_known_classes: int) -> torch.Tensor:
        """Eq. 12: hinge penalty above the max-softmax threshold."""
        logits = self.fc(unseen_z)[:, :num_known_classes]
        confidence = F.softmax(logits, dim=-1).max(dim=-1).values
        return  unseen_z.shape[0] ** 0.5 * 1.5 * F.relu(confidence - self.unseen_threshold).mean()
