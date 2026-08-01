from .afc import afc_nca_loss, afc_pod_loss, scheduled_afc_factor
from .casper import casper_spectral_loss, neural_knn_weights
from .create import (
    ClassAutoencoder,
    ClasswiseAutoencoderClassifier,
    create_classification_loss,
    create_contrastive_loss,
    reconstruction_confidence_weights,
)
from .cscct import (
    controlled_transfer_loss,
    cosine_similarity_matrix,
    cross_space_clustering_loss,
)
from .fgp import (
    RectifiedCosineLinear,
    fgp_graph_preservation_loss,
    pairwise_squared_euclidean,
    scheduled_fgp_weight,
)
from .global_hap import AnchorGeometryLoss
from .icarl import (
    icarl_bce_loss,
    icarl_distillation_targets,
    parameter_l2_regularization,
)
from .logit_kd import old_logit_kl_loss
from .podnet import (
    pod_flat_loss,
    pod_spatial_loss,
    podnet_nca_loss,
)
from .prototype_ce import prototype_cross_entropy, prototype_logits
from .replay_ce import (
    method_uses_afc,
    method_uses_dual_rebalancing,
    method_uses_geometry,
    method_uses_logit_kd,
    method_uses_topkd,
    method_uses_takp,
)
from .sacil_v0 import (
    ConflictWeights,
    compute_conflict_weights,
    global_preservation_weights,
)
from .topology_kd import (
    RipsNet,
    TopologyDistillationLoss,
    load_frozen_ripsnet,
)
from .takp import takp_mixed_classification_loss

__all__ = [
    "AnchorGeometryLoss",
    "ClassAutoencoder",
    "ClasswiseAutoencoderClassifier",
    "ConflictWeights",
    "RectifiedCosineLinear",
    "RipsNet",
    "TopologyDistillationLoss",
    "afc_nca_loss",
    "afc_pod_loss",
    "casper_spectral_loss",
    "compute_conflict_weights",
    "controlled_transfer_loss",
    "cosine_similarity_matrix",
    "create_classification_loss",
    "create_contrastive_loss",
    "cross_space_clustering_loss",
    "fgp_graph_preservation_loss",
    "global_preservation_weights",
    "icarl_bce_loss",
    "icarl_distillation_targets",
    "parameter_l2_regularization",
    "load_frozen_ripsnet",
    "method_uses_afc",
    "method_uses_dual_rebalancing",
    "method_uses_geometry",
    "method_uses_logit_kd",
    "method_uses_topkd",
    "method_uses_takp",
    "old_logit_kl_loss",
    "neural_knn_weights",
    "pairwise_squared_euclidean",
    "pod_flat_loss",
    "pod_spatial_loss",
    "podnet_nca_loss",
    "prototype_cross_entropy",
    "prototype_logits",
    "reconstruction_confidence_weights",
    "scheduled_afc_factor",
    "scheduled_fgp_weight",
    "takp_mixed_classification_loss",
]
