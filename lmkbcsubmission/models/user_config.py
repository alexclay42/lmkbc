from enum import Enum

from models.baseline_qwen_3_model import Qwen3Model


class Models(Enum):
    BASELINE_QWEN_3 = "baseline_qwen_3"

    # Add more models here

    @staticmethod
    def get_model(model_name: str):
        model = Models(model_name)
        if model == Models.BASELINE_QWEN_3:
            return Qwen3Model
        else:
            raise ValueError(f"Model `{model_name}` not found.")
