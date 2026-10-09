from .increfa_base import IncreFABaseClassifier


def load_model(config):
    model = config['model']
    return IncreFABaseClassifier(
        backbone=model.get('backbone', 'ViT-L/14'),
        feature_layer=model.get('feature_layer', 'layer11'),
        device=config['device'],
        num_predictions=config['data']['total_classes'],
        input_dim=model.get('input_dim', 1024),
        hidden_units=model.get('hidden_units', 128),
        num_families=model.get('num_families', 4),
        buffer_size_per_task=model.get('buffer_size', 150),
        unseen_threshold=config.get('unseen_threshold', 0.65),
        fine_loss_weight=config.get('fine_loss_weight', 0.2),
        coarse_loss_weight=config.get('coarse_loss_weight', 0.5),
        unseen_beta_min=config.get('unseen_beta_min', 0.0),
        unseen_beta_max=config.get('unseen_beta_max', 1.0),
    ).to(config['device'])
