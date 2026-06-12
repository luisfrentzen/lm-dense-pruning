from pathlib import Path


class BaseModelAdapter:
    name = "base"

    def matches(self, model_dir: Path, checkpoint_manager) -> bool:
        raise NotImplementedError("Subclasses must implement matches().")

    def load_pretrained_model(self, model_dir: Path, cfg, checkpoint_manager):
        raise NotImplementedError("Subclasses must implement load_pretrained_model().")

    def rebuild_custom_model(self, model_dir: Path, cfg, checkpoint_manager):
        raise NotImplementedError("Subclasses must implement rebuild_custom_model().")

    def load_tokenizer(self, model_dir: Path):
        raise NotImplementedError("Subclasses must implement load_tokenizer().")

    def get_text_layers(self, model):
        raise NotImplementedError("Subclasses must implement get_text_layers().")

    def supports_custom_rebuild(self) -> bool:
        return True

    def supports_pretrained_load(self) -> bool:
        return True