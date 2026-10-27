"""
Cargador de configuración para LLM-FineTuner.

Maneja la carga, validación y fusión de configuraciones desde YAML y variables de entorno.
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)


@dataclass
class ModelConfig:
    """Configuración del modelo."""
    base: str = "Qwen/Qwen2.5-7B-Instruct"
    local_path: Optional[str] = None
    output_dir: str = "data/output"
    quantization: str = "nf4"
    device: str = "auto"


@dataclass
class TrainingConfig:
    """Configuración de entrenamiento."""
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 2e-4
    max_seq_length: int = 4096
    optimizer: str = "paged_adamw_8bit"
    weight_decay: float = 0.01
    warmup_steps: int = 100
    gradient_accumulation_steps: int = 1
    multi_gpu: bool = False
    fp16: bool = True
    bf16: bool = False
    tf32: bool = False
    early_stopping_patience: int = 3
    logging_steps: int = 10
    save_steps: int = 100
    eval_steps: int = 100


@dataclass
class DataConfig:
    """Configuración de datos."""
    input_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    min_length: int = 10
    max_length: int = 4096
    num_workers: int = 2
    shuffle: bool = True
    val_split: float = 0.1
    test_split: float = 0.0


@dataclass
class MergeConfig:
    """Configuración de fusión."""
    merge_base_and_adapter: bool = True
    merged_dir: str = "data/output/merged"


@dataclass
class ExportConfig:
    """Configuración de exportación."""
    format: str = "gguf"
    export_dir: str = "data/output/gguf"
    gguf_quantization: str = "int4"


@dataclass
class OllamaConfig:
    """Configuración de Ollama."""
    model_name: str = "finetuned-qwen2.5-7b"
    template: str = "qwen2.5"
    ollama_path: Optional[str] = None
    ollama_dir: str = "data/output/ollama"
    port: int = 11434


@dataclass
class LoggingConfig:
    """Configuración de logging."""
    level: str = "INFO"
    log_dir: str = "data/logs"
    log_file: str = "training.log"
    tensorboard: bool = False
    tensorboard_dir: str = "data/logs/tensorboard"


@dataclass
class HardwareConfig:
    """Configuración de hardware."""
    num_gpus: int = 1
    gpu_id: int = 0
    max_memory_per_gpu: Optional[int] = None
    cpu_only: bool = False


@dataclass
class ResumeConfig:
    """Configuración de reanudación."""
    from_checkpoint: Optional[str] = None
    save_checkpoint: bool = True
    checkpoint_dir: str = "data/output/checkpoints"
    checkpoint_freq: int = 1


@dataclass
class Config:
    """Configuración completa de LLM-FineTuner."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    merge: MergeConfig = field(default_factory=MergeConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    resume: ResumeConfig = field(default_factory=ResumeConfig)


def load_config(config_path: Optional[str] = None) -> Config:
    """
    Cargar configuración desde archivo YAML.
    
    Args:
        config_path: Ruta al archivo de configuración. Si None, usa "config/config.yaml"
        
    Returns:
        Configuración cargada como objeto Config
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si la configuración es inválida
    """
    if config_path is None:
        config_path = "config/config.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Cargar YAML
    with open(config_path, 'r', encoding='utf-8') as f:
        config_dict = yaml.safe_load(f) or {}
    
    # Validar y convertir a dataclass
    return Config(**config_dict)


def load_config_dict(config_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Cargar configuración como diccionario (para compatibilidad).
    
    Args:
        config_path: Ruta al archivo de configuración
        
    Returns:
        Configuración como diccionario
    """
    if config_path is None:
        config_path = "config/config.yaml"
    
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f) or {}


def validate_config(config: Config) -> List[str]:
    """
    Validar la configuración.
    
    Args:
        config: Configuración a validar
        
    Returns:
        Lista de errores (vacía si válida)
    """
    errors = []
    
    # Validar modelo
    if not config.model.base:
        errors.append("model.base is required")
    
    # Validar cuantización
    valid_quants = ["nf4", "fp4", "int8", "int4"]
    if config.model.quantization not in valid_quants:
        errors.append(f"model.quantization must be one of {valid_quants}")
    
    # Validar dispositivos
    valid_devices = ["auto", "cuda", "cpu"]
    if config.model.device not in valid_devices:
        errors.append(f"model.device must be one of {valid_devices}")
    
    # Validar parámetros numéricos
    if config.training.lora_rank <= 0:
        errors.append("training.lora_rank must be > 0")
    
    if config.training.lora_alpha <= 0:
        errors.append("training.lora_alpha must be > 0")
    
    if config.training.epochs <= 0:
        errors.append("training.epochs must be > 0")
    
    if config.training.batch_size <= 0:
        errors.append("training.batch_size must be > 0")
    
    if config.training.learning_rate <= 0:
        errors.append("training.learning_rate must be > 0")
    
    # Validar directorios
    if not config.data.input_dir:
        errors.append("data.input_dir is required")
    
    if not config.data.processed_dir:
        errors.append("data.processed_dir is required")
    
    return errors


def get_config_from_env() -> Dict[str, Any]:
    """
    Obtener configuración desde variables de entorno.
    
    Las variables de entorno deben tener el prefijo LLMFT_ (ej: LLMFT_MODEL_BASE)
    
    Returns:
        Diccionario con configuración desde entorno
    """
    config = {}
    
    for key, value in os.environ.items():
        if key.startswith("LLMFT_"):
            # Convertir LLMFT_MODEL_BASE a model.base
            path = key[6:].lower().replace("__", ".")
            
            # Intentar convertir tipos
            try:
                if value.lower() in ["true", "false"]:
                    value = value.lower() == "true"
                elif value.isdigit():
                    value = int(value)
                elif value.replace(".", "").replace("-", "").isdigit():
                    value = float(value)
            except (ValueError, AttributeError):
                pass
            
            # Crear estructura anidada
            parts = path.split(".")
            current = config
            for part in parts[:-1]:
                if part not in current:
                    current[part] = {}
                current = current[part]
            current[parts[-1]] = value
    
    return config


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Fusionar dos configuraciones (override tiene prioridad).
    
    Args:
        base: Configuración base
        override: Configuración que sobrescribe
        
    Returns:
        Configuración fusionada
    """
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = value
    
    return result


def load_config_with_env(config_path: Optional[str] = None) -> Config:
    """
    Cargar configuración combinando YAML y variables de entorno.
    
    Args:
        config_path: Ruta al archivo YAML
        
    Returns:
        Configuración combinada
    """
    # Cargar desde YAML
    yaml_config = load_config_dict(config_path)
    
    # Cargar desde entorno
    env_config = get_config_from_env()
    
    # Fusionar
    merged = merge_configs(yaml_config, env_config)
    
    # Validar
    config = Config(**merged)
    errors = validate_config(config)
    if errors:
        for error in errors:
            logger.warning(f"Config validation: {error}")
    
    return config
