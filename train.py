import argparse
import gc
import json
import logging
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.optim import Adam
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from data.forgery_data_arrow import ForgeryArrowDataset
from logger import MixLogger
from models import load_model


class IncreTrainer:

    def __init__(self) -> None:
        self.args = self.parse_args()
        self.config = self.load_config()

        self.logger = MixLogger(self.config['exp'])
        logging.info(self.config)

        self.model = load_model(config=self.config)

        self.setseed()
        self.criterion = torch.nn.CrossEntropyLoss()

        self.dataset = self.load_dataset()
        self.configure_predefined_families()

        total_classes = len(self.dataset.all_classes)
        config_total = self.config['data']['total_classes']
        if config_total != total_classes:
            raise ValueError(
                f"Configured {config_total} known classes, found {total_classes}"
            )
        self.config['data']['total_classes'] = total_classes
        num_tasks = len(self.dataset.order)
        self.metrics = {
            'ACC': torch.ones(num_tasks, total_classes),
            'RaF': [],
            'task_accuracy': [],
            'unseen_as_fake_rate': [],
        }

        self.optimizer, self.scheduler = self.load_optimizer()
        self.global_step = 0

    def configure_predefined_families(self) -> None:
        mapping_path = self.config['model']['family_mapping_path']

        with open(mapping_path, 'r') as stream:
            payload = json.load(stream)
        families = payload.get('families')
        if not isinstance(families, dict) or not families:
            raise ValueError(
                f"{mapping_path} must contain a non-empty 'families' object"
            )
        if len(families) > self.model.num_families:
            raise ValueError(
                f"Mapping defines {len(families)} families, but model supports "
                f"{self.model.num_families}"
            )

        label_to_family = {}
        family_names = {}
        for family_id, (family_name, labels) in enumerate(families.items()):
            family_names[family_id] = family_name
            if not isinstance(labels, list) or not labels:
                raise ValueError(f"Family '{family_name}' must be a non-empty list")
            for label in labels:
                if label in label_to_family:
                    raise ValueError(f"Label '{label}' appears in multiple families")
                label_to_family[label] = family_id

        dataset_labels = set(self.dataset.class_names)
        unknown = sorted(set(label_to_family) - dataset_labels)
        missing = sorted(dataset_labels - set(label_to_family))
        if unknown or missing:
            raise ValueError(
                f"Invalid predefined family mapping; unknown={unknown}, missing={missing}"
            )

        mapped_class_to_family = {}
        for original_id in self.dataset.all_classes:
            label = self.dataset.idx_to_label_str[original_id]
            mapped_id = self.dataset.label_mapping[original_id]
            mapped_class_to_family[mapped_id] = label_to_family[label]
        self.model.set_class_family_mapping(mapped_class_to_family, family_names)
        logging.info("Validated predefined family mapping %s", mapping_path)

    def parse_args(self):
        parser = argparse.ArgumentParser(description='IncreFA')
        parser.add_argument('--config', default='configs/increfa.yml')
        return parser.parse_args()

    def setseed(self):
        torch.manual_seed(self.config['seed'])
        torch.cuda.manual_seed(self.config['seed'])
        np.random.seed(self.config['seed'])
        random.seed(self.config['seed'])

    def load_optimizer(self):
        optimizer = Adam(
            self.model.parameters(), lr=self.config['optimizer']['base_lr']
        )
        scheduler = PolynomialLR(
            optimizer,
            total_iters=self.config['train']['epochs'],
            power=self.config['optimizer']['scheduler_power'],
        )
        return optimizer, scheduler

    def load_config(self) -> dict:
        with open(self.args.config, 'r') as f:
            return yaml.safe_load(f)

    def load_dataset(self):
        data = self.config['data']
        return ForgeryArrowDataset(
            arrow_data_dir=data['arrow_data_dir'],
            real_label_name=data['real_label_name'],
            task_sizes=data['task_sizes'],
            time_series=data['time_series'],
            holdout_labels=data['holdout_labels'],
            split_ratio=data['split_ratio'],
            split_seed=data['split_seed'],
            split_counts=data['split_counts'],
        )

    def cleanup_memory(self):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _configure_task_scheduler(self, task_id: int, max_epoch: int) -> None:
        task_lr = self.config['optimizer']['base_lr']
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = task_lr
            param_group['initial_lr'] = task_lr
        self.scheduler = PolynomialLR(
            self.optimizer,
            total_iters=max_epoch,
            power=self.config['optimizer']['scheduler_power'],
        )
        logging.info(
            f"Task {task_id} learning rate: {task_lr:.8f} "
            f"(epochs={max_epoch})"
        )

    def start(self) -> None:
        num_learned_classes: int = 0

        for task_id, (train_dataset, test_dataset, unseen_test_dataset, meta) in enumerate(
                self.dataset.incremental_learning()):

            num_learning_classes = len(meta['current_classes'])
            self.model.orth_space.update_indicator(
                self.model.orth_space.learning_indicator + num_learning_classes
            )
            logging.info(meta)

            self.model.replay_buffer.set_current_task(task_id)

            train_loader = DataLoader(
                dataset=train_dataset,
                batch_size=self.config['data']['batch_size'],
                num_workers=self.config['data']['num_workers'],
                pin_memory=self.config['data']['pin_memory'],
                shuffle=True,
            )

            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.config['data']['batch_size'],
                num_workers=self.config['data']['num_workers'],
                pin_memory=self.config['data']['pin_memory'],
            )

            max_epoch = self.config['train']['epochs']
            self._configure_task_scheduler(task_id, max_epoch)

            for epoch in range(max_epoch):
                self.model.train()
                y_gt_all, y_pred_all = [], []

                unseen_mix_weight = self.config['unseen_mix_weight']
                unseen_batch_size = self.config['unseen_batch_size']
                unseen_num_mix = self.config['unseen_num_mix']
                unseen_current_mix_fraction = self.config.get(
                    'unseen_current_mix_fraction', 0.0,
                )
                orth_loss_weight = self.config['orth_loss_weight']
                replay_loss_weight = self.config.get('replay_loss_weight', 1.0)
                active_classes = num_learned_classes + num_learning_classes

                for X, y_gt in tqdm(train_loader, desc=f"Task {task_id} Epoch {epoch}"):
                    y_gt = self.dataset.apply_label_mapping_batch(y_gt)
                    X = X.to(self.config['device'])
                    y_gt_dev = y_gt.to(self.config['device'])

                    y_pred, orth_loss = self.model(
                        X, y_gt_dev, store_samples=(epoch == 0)
                    )

                    y_pred = y_pred[:, :active_classes]

                    current_loss = self.criterion(y_pred, y_gt_dev) + orth_loss_weight * orth_loss
                    loss = current_loss

                    replay_loss_val = 0.0
                    unseen_loss_val = 0.0
                    current_loss_val = current_loss.detach().cpu().item()
                    current_unique_classes = y_gt_dev.unique().numel()
                    replay_unique_classes = 0
                    replay_max_class_fraction = 0.0
                    if task_id > 0:
                        replay_batch_size = self.config['data'].get('replay_batch_size',
                                                                     self.config['data']['batch_size'])
                        replay_x, replay_y = self.model.get_replay_batch(batch_size=replay_batch_size)
                        if replay_x is not None:
                            replay_counts = replay_y.unique(return_counts=True)[1]
                            replay_unique_classes = replay_counts.numel()
                            replay_max_class_fraction = (
                                replay_counts.max().float() / replay_y.numel()
                            ).item()
                            replay_pred, replay_orth = self.model.replay_forward(replay_x)
                            replay_pred = replay_pred[:, :active_classes]
                            replay_loss = self.criterion(replay_pred, replay_y) + replay_orth
                            loss = loss + replay_loss_weight * replay_loss
                            replay_loss_val = replay_loss.detach().cpu().item()

                        unseen_kwargs = {
                            'batch_size': unseen_batch_size,
                            'num_mix': unseen_num_mix,
                            'current_mix_fraction': unseen_current_mix_fraction,
                        }
                        unseen_features = self.model.generate_unseen_samples(
                            **unseen_kwargs
                        )
                        if unseen_features is not None:
                            unseen_loss = self.model.compute_unseen_loss(
                                unseen_features, active_classes,
                            )
                            loss = loss + unseen_mix_weight * unseen_loss
                            unseen_loss_val = unseen_loss.detach().cpu().item()

                    loss.backward()
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                    if self.global_step % 10 == 0:
                        log_dict = {
                            "train/total_loss": loss.detach().cpu().item(),
                            "train/current_loss": current_loss_val,
                            "train/orth_loss": orth_loss.detach().cpu().item(),
                            "train/replay_loss": replay_loss_val,
                            "train/unseen_loss": unseen_loss_val,
                            "train/current_unique_classes": current_unique_classes,
                            "train/replay_unique_classes": replay_unique_classes,
                            "train/replay_max_class_fraction": replay_max_class_fraction,
                            "train/unseen_current_mix_fraction": unseen_current_mix_fraction,
                        }
                        self.logger.log_scalars(log_dict, self.global_step)

                    pred_labels = torch.argmax(y_pred, dim=1)
                    y_gt_all.append(y_gt.cpu())
                    y_pred_all.append(pred_labels.detach().cpu())
                    self.global_step += 1

                y_pred_all = torch.cat(y_pred_all, dim=0)
                y_gt_all = torch.cat(y_gt_all, dim=0)
                train_acc = (y_gt_all == y_pred_all).float().mean() * 100
                logging.info(f"Training Acc: {train_acc:.2f}%")

                used_lr = self.optimizer.param_groups[0]['lr']
                self.scheduler.step()
                self.logger.log_scalars({
                    "train/learning_rate": used_lr,
                    "train/next_learning_rate": self.optimizer.param_groups[0]['lr'],
                }, self.global_step)
                del y_pred_all, y_gt_all
                self.cleanup_memory()

            ood_threshold, seen_logits = self.test(task_id, test_loader,
                                      num_learning_classes, num_learned_classes)

            if unseen_test_dataset is not None:
                unseen_loader = DataLoader(
                    dataset=unseen_test_dataset,
                    batch_size=self.config['data']['batch_size'],
                    num_workers=self.config['data']['num_workers'],
                    pin_memory=self.config['data']['pin_memory'],
                )
                self.test_unseen(task_id, unseen_loader,
                                 num_learned_classes + num_learning_classes,
                                 ood_threshold, seen_logits)
                del unseen_loader

            self.model.finalize_task(task_id)
            num_learned_classes += num_learning_classes
            self.save_progress(task_id, num_learned_classes)
            del train_loader, test_loader
            self.cleanup_memory()

    def save_progress(self, task_id: int, num_learned_classes: int) -> None:
        output_dir = os.path.join('outputs', self.config['exp'])
        os.makedirs(output_dir, exist_ok=True)

        results = {
            'task_id': task_id,
            'global_step': self.global_step,
            'num_learned_classes': num_learned_classes,
            'task_accuracy': self.metrics['task_accuracy'],
            'auth_accuracy': self.metrics['RaF'],
            'unseen_accuracy': [value * 100 for value in self.metrics['unseen_as_fake_rate']],
            'class_order': [
                [self.dataset.idx_to_label_str[class_id] for class_id in task]
                for task in self.dataset.order
            ],
            'heldout_classes': [
                self.dataset.idx_to_label_str[class_id]
                for class_id in self.dataset.heldout_classes
            ],
        }
        results_tmp = os.path.join(output_dir, 'table1_results.json.tmp')
        results_path = os.path.join(output_dir, 'table1_results.json')
        with open(results_tmp, 'w') as stream:
            json.dump(results, stream, indent=2)
        os.replace(results_tmp, results_path)

        trainable_state = {
            key: value.cpu()
            for key, value in self.model.state_dict().items()
            if not key.startswith('backbone.')
        }
        checkpoint = {
            'task_id': task_id,
            'global_step': self.global_step,
            'num_learned_classes': num_learned_classes,
            'model': trainable_state,
            'optimizer': self.optimizer.state_dict(),
            'replay': self.model.replay_buffer.task_tensors,
            'results': results,
            'config': self.config,
        }
        checkpoint['class_to_family'] = self.model.class_to_family
        checkpoint_tmp = os.path.join(output_dir, 'latest.pt.tmp')
        checkpoint_path = os.path.join(output_dir, 'latest.pt')
        torch.save(checkpoint, checkpoint_tmp)
        os.replace(checkpoint_tmp, checkpoint_path)

    def test(self, task_id: int, test_loader: DataLoader,
             num_learning_classes: int, num_learned_classes: int) -> tuple[float, torch.Tensor]:
        self.model.eval()
        y_gt_all, y_pred_argmax_all = [], []
        seen_logits_all = []
        active_classes = num_learned_classes + num_learning_classes

        with torch.no_grad():
            for X, y_gt in tqdm(test_loader, desc=f"Testing task {task_id}"):
                y_gt = self.dataset.apply_label_mapping_batch(y_gt)
                X = X.to(self.config['device'])

                predictions, _ = self.model(X, y_gt=None, store_samples=False)
                logits = predictions[:, :active_classes]

                argmax_pred = torch.argmax(logits, dim=1)
                y_pred_argmax_all.append(argmax_pred.detach().cpu())
                y_gt_all.append(y_gt.cpu())

                seen_logits_all.append(logits.detach().cpu())

        y_pred_argmax_all = torch.cat(y_pred_argmax_all, dim=0)
        y_gt_all = torch.cat(y_gt_all, dim=0)
        self.calculate_metrics(task_id, y_pred_argmax_all, y_gt_all,
                               active_classes)

        seen_logits = torch.cat(seen_logits_all, dim=0)
        seen_confidence = F.softmax(seen_logits, dim=-1).max(dim=-1).values
        auto_threshold = self.config.get('unseen_threshold', 0.65)
        logging.info(
            f"OOD max-softmax threshold={auto_threshold:.4f}, "
            f"known confidence mean={seen_confidence.mean():.4f}, "
            f"std={seen_confidence.std():.4f}"
        )

        self.logger.log_scalars({
            "ood/known_confidence_mean": seen_confidence.mean().item(),
            "ood/known_confidence_std": seen_confidence.std().item(),
            "ood/threshold": auto_threshold,
        }, self.global_step)

        del y_pred_argmax_all, y_gt_all, seen_logits_all
        self.cleanup_memory()
        return auto_threshold, seen_logits

    @staticmethod
    def _known_vs_unseen_auroc(known_scores: torch.Tensor,
                               unseen_scores: torch.Tensor) -> float:
        """Mann-Whitney AUROC where larger scores indicate known samples."""
        scores = torch.cat([known_scores, unseen_scores]).float()
        order = torch.argsort(scores)
        ranks = torch.empty_like(scores)
        ranks[order] = torch.arange(
            1, scores.numel() + 1, dtype=scores.dtype,
        )
        n_known = known_scores.numel()
        n_unseen = unseen_scores.numel()
        known_rank_sum = ranks[:n_known].sum()
        auc = (
            known_rank_sum - n_known * (n_known + 1) / 2
        ) / (n_known * n_unseen)
        return auc.item()

    def test_unseen(self, task_id: int, unseen_loader: DataLoader,
                    active_classes: int, ood_threshold: float,
                    seen_logits: torch.Tensor = None) -> None:
        self.model.eval()
        real_label = self.dataset.mapped_real_label

        total = 0
        detected_as_unseen = 0
        predicted_as_real = 0
        unseen_logits_all = []
        unseen_labels_all = []

        with torch.no_grad():
            for X, y_original in tqdm(unseen_loader, desc=f"Unseen eval task {task_id}"):
                X = X.to(self.config['device'])

                predictions, _ = self.model(X, y_gt=None, store_samples=False)
                logits = predictions[:, :active_classes]
                unseen_logits_all.append(logits.detach().cpu())
                unseen_labels_all.append(y_original.cpu())
                confidence = F.softmax(logits, dim=-1).max(dim=-1).values
                is_unseen = confidence < ood_threshold
                pred_indices = logits.argmax(dim=-1)
                pred_labels = pred_indices.clone()
                pred_labels[is_unseen] = -1

                total += pred_labels.size(0)
                detected_as_unseen += is_unseen.sum().item()
                predicted_as_real += (pred_labels == real_label).sum().item()

        if total > 0:
            unseen_detection_rate = detected_as_unseen / total
            as_real_rate = predicted_as_real / total

            self.metrics['unseen_as_fake_rate'].append(unseen_detection_rate)

            unseen_logits = torch.cat(unseen_logits_all, dim=0)
            unseen_labels = torch.cat(unseen_labels_all, dim=0)
            unseen_confidence = F.softmax(unseen_logits, dim=-1).max(dim=-1).values
            unseen_predictions = unseen_logits.argmax(dim=-1)
            unseen_max = unseen_logits.max(dim=-1).values
            seen_confidence = None
            seen_max = None
            log_parts = [
                f"Task {task_id} Unseen Detection — "
                f"acc: {unseen_detection_rate * 100:.2f}% ({detected_as_unseen}/{total})",
            ]
            if seen_logits is not None:
                seen_max = seen_logits.max(dim=-1).values
                seen_confidence = F.softmax(seen_logits, dim=-1).max(dim=-1).values
                log_parts.append(
                    f"  Seen logits  — max: mean={seen_max.mean():.4f}, "
                    f"std={seen_max.std():.4f}, min={seen_max.min():.4f}, max={seen_max.max():.4f}"
                )
            log_parts.append(
                f"  Unseen logits — max: mean={unseen_max.mean():.4f}, "
                f"std={unseen_max.std():.4f}, min={unseen_max.min():.4f}, max={unseen_max.max():.4f}"
            )
            top_class, top_count = unseen_predictions.unique(return_counts=True)
            top_pos = top_count.argmax()
            top_mapped_id = top_class[top_pos].item()
            inverse_mapping = {
                mapped: original
                for original, mapped in self.dataset.label_mapping.items()
            }
            top_original_id = inverse_mapping[top_mapped_id]
            top_class_name = self.dataset.idx_to_label_str[top_original_id]
            top_concentration = top_count[top_pos].float() / total
            log_parts.append(
                f"  Unseen top prediction — {top_class_name}: "
                f"{top_concentration * 100:.2f}% ({top_count[top_pos].item()}/{total})"
            )
            for original_id in self.dataset.heldout_classes:
                class_mask = unseen_labels == original_id
                if class_mask.any():
                    class_name = self.dataset.idx_to_label_str[original_id]
                    class_detect = (
                        unseen_confidence[class_mask] < ood_threshold
                    ).float().mean()
                    log_parts.append(
                        f"  Held-out {class_name} — detect={class_detect * 100:.2f}%, "
                        f"MSP mean={unseen_confidence[class_mask].mean():.4f}, "
                        f"median={unseen_confidence[class_mask].median():.4f}"
                    )
            logging.info("\n".join(log_parts))

            ood_metrics = {
                "ood/unseen_detection_rate": unseen_detection_rate,
                "ood/unseen_as_real_rate": as_real_rate,
                "ood/unseen_confidence_mean": unseen_confidence.mean().item(),
                "ood/unseen_confidence_median": unseen_confidence.median().item(),
                "ood/unseen_top1_concentration": top_concentration.item(),
                "ood/unseen_top1_class_id": top_mapped_id,
            }
            if seen_confidence is not None:
                ood_metrics.update({
                    "ood/seen_false_unseen_rate": (
                        seen_confidence < ood_threshold
                    ).float().mean().item(),
                    "ood/msp_auroc": self._known_vs_unseen_auroc(
                        seen_confidence, unseen_confidence,
                    ),
                    "ood/maxlogit_auroc": self._known_vs_unseen_auroc(
                        seen_max, unseen_max,
                    ),
                })
            for original_id in self.dataset.heldout_classes:
                class_mask = unseen_labels == original_id
                if class_mask.any():
                    class_name = self.dataset.idx_to_label_str[original_id]
                    ood_metrics[f"ood/heldout_{class_name}_detection_rate"] = (
                        unseen_confidence[class_mask] < ood_threshold
                    ).float().mean().item()
                    ood_metrics[f"ood/heldout_{class_name}_confidence_mean"] = (
                        unseen_confidence[class_mask].mean().item()
                    )
            self.logger.log_scalars(ood_metrics, self.global_step)

            del unseen_logits_all, unseen_labels_all

    def calculate_metrics(self, task_id: int, preds: torch.Tensor,
                          gts: torch.Tensor, active_classes: int):
        total_classes = self.config['data']['total_classes']

        accuracy = (preds == gts).float().mean() * 100

        real_label = self.dataset.mapped_real_label
        gt_is_real = (gts == real_label)
        pred_is_real = (preds == real_label)

        real_total = gt_is_real.sum()
        real_recall = (gt_is_real & pred_is_real).float().sum() / real_total if real_total > 0 else torch.tensor(0.0)

        gt_is_fake = ~gt_is_real
        pred_is_fake = ~pred_is_real
        fake_total = gt_is_fake.sum()
        fake_recall = (gt_is_fake & pred_is_fake).float().sum() / fake_total if fake_total > 0 else torch.tensor(0.0)

        raf_acc = (real_recall + fake_recall) / 2.0 * 100

        logging.info(
            f"Task {task_id} — Accuracy: {accuracy:.2f}%, "
            f"RaF: {raf_acc:.2f}% (real_recall={real_recall*100:.2f}%, "
            f"fake_recall={fake_recall*100:.2f}%)"
        )

        inv_mapping = {v: k for k, v in self.dataset.label_mapping.items()}
        class_acc_logs = []

        for i in range(total_classes):
            class_mask = (gts == i)
            if i < active_classes and class_mask.sum() > 0:
                class_acc = (preds[class_mask] == i).float().mean() * 100
                self.metrics['ACC'][task_id, i] = class_acc
                
                orig_lbl_idx = inv_mapping.get(i, i)
                class_name = self.dataset.idx_to_label_str.get(orig_lbl_idx, f"class_{orig_lbl_idx}")
                
                class_acc_logs.append(f"{class_name}: {class_acc:.2f}%")
            else:
                self.metrics['ACC'][task_id, i] = float('nan')

        if class_acc_logs:
            logging.info(f"Task {task_id} Per-class Accuracy — " + ", ".join(class_acc_logs))

        self.metrics['RaF'].append(raf_acc.item())
        self.metrics['task_accuracy'].append(accuracy.item())

        logs = {
            "test/overall_accuracy": accuracy.item(),
            "test/raf_accuracy": raf_acc.item(),
            "test/real_recall": real_recall.item() * 100,
            "test/fake_recall": fake_recall.item() * 100,
            "test/task_id": task_id,
        }
        
        for i in range(active_classes):
            val = self.metrics['ACC'][task_id, i]
            if not torch.isnan(val):
                orig_lbl_idx = inv_mapping.get(i, i)
                class_name = self.dataset.idx_to_label_str.get(orig_lbl_idx, f"class_{orig_lbl_idx}")
                logs[f"test/class_{class_name}_accuracy"] = val.item()

        buffer_status = self.model.get_buffer_status()
        logs["replay_buffer/total_samples"] = buffer_status['total_samples']
        logs["replay_buffer/current_task"] = buffer_status['current_task']

        self.logger.log_scalars(logs, self.global_step)


if __name__ == '__main__':
    trainer = IncreTrainer()
    trainer.start()
