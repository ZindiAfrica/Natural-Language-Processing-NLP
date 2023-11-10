from transformers.models.auto.configuration_auto import CONFIG_MAPPING_NAMES
from transformers.models.auto.modeling_auto import (
    MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES,
    _LazyAutoMapping,
)

from .modeling_m2m_100 import M2M100ForTokenClassification

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES["m2m_100"] = (
    "M2M100ForTokenClassification"
)

MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING = _LazyAutoMapping(
    CONFIG_MAPPING_NAMES, MODEL_FOR_TOKEN_CLASSIFICATION_MAPPING_NAMES
)

from transformers.models import m2m_100

setattr(m2m_100, "M2M100ForTokenClassification", M2M100ForTokenClassification)

__version__ = "0.1"

__all__ = [
    "M2M100ForTokenClassification",
]
