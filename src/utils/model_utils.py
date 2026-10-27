"""
Utilidades de modelos para LLM-FineTuner.

Proporciona funciones para manejar modelos de lenguaje, validación y estimación de recursos.
"""

import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


class ModelArchitecture(Enum):
    """Arquitecturas de modelos soportadas."""
    QWEN = "qwen"
    QWEN2 = "qwen2"
    QWEN2_5 = "qwen2.5"
    MISTRAL = "mistral"
    LLAMA = "llama"
    GEMMA = "gemma"
    UNKNOWN = "unknown"


class QuantizationType(Enum):
    """Tipos de cuantización soportados."""
    NONE = "none"
    FP16 = "float16"
    BF16 = "bfloat16"
    FP32 = "float32"
    INT8 = "int8"
    INT4 = "int4"
    NF4 = "nf4"
    FP4 = "fp4"


@dataclass
class ModelInfo:
    """Información sobre un modelo."""
    name: str
    architecture: ModelArchitecture
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: Optional[int] = None
    vocab_size: int = 0
    max_position_embeddings: int = 0
    parameter_count: int = 0
    
    @property
    def size_category(self) -> str:
        """Categoría de tamaño del modelo."""
        if self.parameter_count < 1e9:
            return "<1B"
        elif self.parameter_count < 3e9:
            return "1B-3B"
        elif self.parameter_count < 8e9:
            return "3B-7B"
        elif self.parameter_count < 15e9:
            return "7B-14B"
        elif self.parameter_count < 35e9:
            return "14B-32B"
        elif self.parameter_count < 75e9:
            return "32B-70B"
        else:
            return ">70B"


@dataclass
class VRAMRequirements:
    """Requisitos de VRAM estimados."""
    model_size_gb: float
    lora_size_gb: float
    total_required_gb: float
    recommended_gb: float
    quantization: QuantizationType
    batch_size: int = 1
    sequence_length: int = 4096
    
    def to_dict(self) -> Dict:
        """Convertir a diccionario."""
        return {
            "model_size_gb": round(self.model_size_gb, 2),
            "lora_size_gb": round(self.lora_size_gb, 2),
            "total_required_gb": round(self.total_required_gb, 2),
            "recommended_gb": round(self.recommended_gb, 2),
            "quantization": self.quantization.value,
            "batch_size": self.batch_size,
            "sequence_length": self.sequence_length
        }


# Tamaños aproximados por parametro (en bytes) para diferentes cuantizaciones
PARAMETER_SIZES = {
    QuantizationType.FP32: 4,
    QuantizationType.FP16: 2,
    QuantizationType.BF16: 2,
    QuantizationType.INT8: 1,
    QuantizationType.INT4: 0.5,
    QuantizationType.NF4: 0.5,
    QuantizationType.FP4: 0.5,
    QuantizationType.NONE: 4,  # Por defecto float32
}

# Tamaño de LoRA por capa (estimación)
LORA_SIZE_PER_LAYER = {
    1: 0.1,   # rank 1
    2: 0.2,
    4: 0.4,
    8: 0.8,
    16: 1.6,
    32: 3.2,
    64: 6.4,
    128: 12.8,
}


def get_model_info(model_name_or_path: str) -> ModelInfo:
    """
    Obtener información sobre un modelo.
    
    Args:
        model_name_or_path: Nombre en HuggingFace o ruta local
        
    Returns:
        ModelInfo con información del modelo
    """
    # Mapeo de modelos conocidos
    KNOWN_MODELS = {
        # Qwen
        "qwen/qwen-1_8b": ModelInfo("Qwen-1.8B", ModelArchitecture.QWEN, 24, 2048, 12, vocab_size=151936, parameter_count=1800000000),
        "qwen/qwen-7b": ModelInfo("Qwen-7B", ModelArchitecture.QWEN, 32, 4096, 16, vocab_size=151936, parameter_count=7700000000),
        "qwen/qwen-14b": ModelInfo("Qwen-14B", ModelArchitecture.QWEN, 40, 5120, 20, vocab_size=151936, parameter_count=14300000000),
        "qwen/qwen-72b": ModelInfo("Qwen-72B", ModelArchitecture.QWEN, 80, 8192, 32, vocab_size=151936, parameter_count=73800000000),
        
        # Qwen2
        "qwen/qwen2-0_5b": ModelInfo("Qwen2-0.5B", ModelArchitecture.QWEN2, 24, 2560, 16, vocab_size=151936, parameter_count=500000000),
        "qwen/qwen2-1_5b": ModelInfo("Qwen2-1.5B", ModelArchitecture.QWEN2, 24, 3072, 16, vocab_size=151936, parameter_count=1500000000),
        "qwen/qwen2-7b": ModelInfo("Qwen2-7B", ModelArchitecture.QWEN2, 32, 4096, 16, vocab_size=151936, parameter_count=7200000000),
        "qwen/qwen2-72b": ModelInfo("Qwen2-72B", ModelArchitecture.QWEN2, 80, 8192, 32, vocab_size=151936, parameter_count=72000000000),
        
        # Qwen2.5
        "qwen/qwen2.5-0.5b": ModelInfo("Qwen2.5-0.5B", ModelArchitecture.QWEN2_5, 24, 2560, 16, vocab_size=151936, parameter_count=500000000),
        "qwen/qwen2.5-1.5b": ModelInfo("Qwen2.5-1.5B", ModelArchitecture.QWEN2_5, 24, 3072, 16, vocab_size=151936, parameter_count=1500000000),
        "qwen/qwen2.5-3b": ModelInfo("Qwen2.5-3B", ModelArchitecture.QWEN2_5, 32, 4096, 16, vocab_size=151936, parameter_count=3000000000),
        "qwen/qwen2.5-7b": ModelInfo("Qwen2.5-7B", ModelArchitecture.QWEN2_5, 32, 4096, 16, vocab_size=151936, parameter_count=7200000000),
        "qwen/qwen2.5-14b": ModelInfo("Qwen2.5-14B", ModelArchitecture.QWEN2_5, 40, 5120, 20, vocab_size=151936, parameter_count=14000000000),
        "qwen/qwen2.5-32b": ModelInfo("Qwen2.5-32B", ModelArchitecture.QWEN2_5, 64, 8192, 32, vocab_size=151936, parameter_count=32000000000),
        "qwen/qwen2.5-72b": ModelInfo("Qwen2.5-72B", ModelArchitecture.QWEN2_5, 80, 8192, 32, vocab_size=151936, parameter_count=72000000000),
        
        # Mistral
        "mistralai/mistral-7b": ModelInfo("Mistral-7B", ModelArchitecture.MISTRAL, 32, 4096, 32, num_key_value_heads=8, vocab_size=32000, parameter_count=7400000000),
        "mistralai/mistral-7b-instruct": ModelInfo("Mistral-7B-Instruct", ModelArchitecture.MISTRAL, 32, 4096, 32, num_key_value_heads=8, vocab_size=32000, parameter_count=7400000000),
        "mistralai/mixtral-8x7b": ModelInfo("Mixtral-8x7B", ModelArchitecture.MISTRAL, 32, 4096, 32, num_key_value_heads=8, vocab_size=32000, parameter_count=47000000000),
        "mistralai/mixtral-8x22b": ModelInfo("Mixtral-8x22B", ModelArchitecture.MISTRAL, 70, 65536, 64, num_key_value_heads=8, vocab_size=141184, parameter_count=149000000000),
        
        # Llama
        "meta-llama/llama-2-7b": ModelInfo("Llama2-7B", ModelArchitecture.LLAMA, 32, 4096, 32, num_key_value_heads=32, vocab_size=32000, parameter_count=7000000000),
        "meta-llama/llama-2-13b": ModelInfo("Llama2-13B", ModelArchitecture.LLAMA, 40, 5120, 40, num_key_value_heads=40, vocab_size=32000, parameter_count=13000000000),
        "meta-llama/llama-2-70b": ModelInfo("Llama2-70B", ModelArchitecture.LLAMA, 80, 8192, 64, num_key_value_heads=8, vocab_size=32000, parameter_count=70000000000),
        
        # Llama3
        "meta-llama/meta-llama-3-8b": ModelInfo("Llama3-8B", ModelArchitecture.LLAMA, 32, 8192, 32, num_key_value_heads=8, vocab_size=128256, parameter_count=8100000000),
        "meta-llama/meta-llama-3-70b": ModelInfo("Llama3-70B", ModelArchitecture.LLAMA, 80, 8192, 64, num_key_value_heads=8, vocab_size=128256, parameter_count=70000000000),
        
        # Gemma
        "google/gemma-2b": ModelInfo("Gemma-2B", ModelArchitecture.GEMMA, 18, 2560, 8, num_key_value_heads=8, vocab_size=256000, parameter_count=2500000000),
        "google/gemma-7b": ModelInfo("Gemma-7B", ModelArchitecture.GEMMA, 28, 4096, 16, num_key_value_heads=8, vocab_size=256000, parameter_count=9000000000),
    }
    
    # Normalizar nombre
    model_name = model_name_or_path.lower()
    
    # Buscar en modelos conocidos
    for key, info in KNOWN_MODELS.items():
        if key in model_name:
            return info
    
    # Intentar cargar config.json localmente
    model_path = Path(model_name_or_path)
    config_file = model_path / "config.json"
    if config_file.exists():
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            num_layers = config.get("num_hidden_layers", 0)
            hidden_size = config.get("hidden_size", 0)
            num_attention_heads = config.get("num_attention_heads", 0)
            vocab_size = config.get("vocab_size", 0)
            
            # Estimar parámetros
            # Para transformer: params ≈ num_layers * (12 * hidden_size^2 + 13 * hidden_size * vocab_size)
            # Esto es una simplificación
            parameter_count = 0
            if num_layers > 0 and hidden_size > 0:
                # Estimación aproximada
                parameter_count = num_layers * hidden_size * num_attention_heads * 4
            
            # Detectar arquitectura
            architecture = ModelArchitecture.UNKNOWN
            if "qwen" in model_name:
                if "2.5" in model_name:
                    architecture = ModelArchitecture.QWEN2_5
                elif "2" in model_name:
                    architecture = ModelArchitecture.QWEN2
                else:
                    architecture = ModelArchitecture.QWEN
            elif "mistral" in model_name:
                architecture = ModelArchitecture.MISTRAL
            elif "llama" in model_name:
                architecture = ModelArchitecture.LLAMA
            elif "gemma" in model_name:
                architecture = ModelArchitecture.GEMMA
            
            return ModelInfo(
                name=model_name_or_path,
                architecture=architecture,
                num_layers=num_layers,
                hidden_size=hidden_size,
                num_attention_heads=num_attention_heads,
                vocab_size=vocab_size,
                parameter_count=parameter_count
            )
        except Exception as e:
            logger.warning(f"Error reading config.json: {e}")
    
    # Si no se encuentra, devolver modelo genérico
    logger.warning(f"Model info not found for '{model_name_or_path}', returning generic info")
    
    # Intentar detectar arquitectura desde el nombre
    architecture = ModelArchitecture.UNKNOWN
    if "qwen" in model_name:
        if "2.5" in model_name:
            architecture = ModelArchitecture.QWEN2_5
        elif "2" in model_name:
            architecture = ModelArchitecture.QWEN2
        else:
            architecture = ModelArchitecture.QWEN
    elif "mistral" in model_name:
        architecture = ModelArchitecture.MISTRAL
    elif "llama" in model_name:
        architecture = ModelArchitecture.LLAMA
    elif "gemma" in model_name:
        architecture = ModelArchitecture.GEMMA
    
    return ModelInfo(
        name=model_name_or_path,
        architecture=architecture,
        num_layers=0,
        hidden_size=0,
        num_attention_heads=0,
        parameter_count=0
    )


def detect_model_architecture(model_name_or_path: str) -> ModelArchitecture:
    """
    Detectar arquitectura de un modelo.
    
    Args:
        model_name_or_path: Nombre o ruta del modelo
        
    Returns:
        Arquitectura detectada
    """
    model_name = model_name_or_path.lower()
    
    if "qwen" in model_name:
        if "2.5" in model_name:
            return ModelArchitecture.QWEN2_5
        elif "2" in model_name:
            return ModelArchitecture.QWEN2
        else:
            return ModelArchitecture.QWEN
    elif "mistral" in model_name or "mixtral" in model_name:
        return ModelArchitecture.MISTRAL
    elif "llama" in model_name:
        return ModelArchitecture.LLAMA
    elif "gemma" in model_name:
        return ModelArchitecture.GEMMA
    else:
        return ModelArchitecture.UNKNOWN


def validate_model_path(model_path: str) -> bool:
    """
    Validar que una ruta de modelo es válida.
    
    Args:
        model_path: Ruta al modelo
        
    Returns:
        True si la ruta es válida
    """
    path = Path(model_path)
    
    # Verificar si es un directorio con config.json
    if path.is_dir():
        config_file = path / "config.json"
        safetensors_file = path / "model.safetensors"
        pytorch_file = path / "pytorch_model.bin"
        
        if config_file.exists() and (safetensors_file.exists() or pytorch_file.exists()):
            return True
        
        # Verificar HuggingFace
        if path.name in ["snapshots", "model"]:
            return True
    
    # Verificar si es un archivo GGUF
    if path.is_file() and path.suffix.lower() == ".gguf":
        return True
    
    return False


def estimate_vram_requirements(
    model_name_or_path: str,
    lora_rank: int = 16,
    quantization: str = "nf4",
    batch_size: int = 1,
    sequence_length: int = 4096,
    num_gpus: int = 1
) -> VRAMRequirements:
    """
    Estimar requisitos de VRAM para entrenamiento.
    
    Args:
        model_name_or_path: Nombre o ruta del modelo base
        lora_rank: Rango LoRA
        quantization: Tipo de cuantización
        batch_size: Tamaño de batch
        sequence_length: Longitud de secuencia
        num_gpus: Número de GPUs
        
    Returns:
        VRAMRequirements con estimaciones
    """
    # Obtener información del modelo
    model_info = get_model_info(model_name_or_path)
    
    # Convertir cuantización
    quant_type = QuantizationType.NF4
    for q in QuantizationType:
        if q.value == quantization.lower():
            quant_type = q
            break
    
    # Calcular tamaño del modelo en GB
    param_size_bytes = PARAMETER_SIZES.get(quant_type, PARAMETER_SIZES[QuantizationType.NONE])
    model_size_bytes = model_info.parameter_count * param_size_bytes
    model_size_gb = model_size_bytes / (1024 ** 3)
    
    # Estimar tamaño de LoRA (depende de la arquitectura)
    # LoRA añade parámetros adicionales por capa
    if lora_rank in LORA_SIZE_PER_LAYER:
        lora_size_per_layer_gb = LORA_SIZE_PER_LAYER[lora_rank]
    else:
        # Interpolación para ranks no definidos
        if lora_rank < 16:
            lora_size_per_layer_gb = LORA_SIZE_PER_LAYER[16] * (lora_rank / 16)
        else:
            lora_size_per_layer_gb = LORA_SIZE_PER_LAYER[16] * (lora_rank / 16)
    
    lora_size_gb = lora_size_per_layer_gb * model_info.num_layers
    
    # Estimación de VRAM requerida (simplificada)
    # VRAM ≈ model_size + lora_size + activations + gradients + overhead
    # Con QLoRA, el modelo base está congelado, así que solo necesitamos:
    # - Modelo base en 4-bit: ~model_size
    # - Adaptadores LoRA: ~lora_size
    # - Activaciones: ~batch_size * sequence_length * hidden_size * 4 bytes
    
    activation_size_bytes = batch_size * sequence_length * model_info.hidden_size * 4
    activation_size_gb = activation_size_bytes / (1024 ** 3)
    
    # Gradientes (para LoRA)
    gradient_size_gb = lora_size_gb * 2  # Estimación
    
    # Overhead
    overhead_gb = 2.0  # 2GB overhead
    
    # Total requerido
    total_required_gb = model_size_gb + lora_size_gb + activation_size_gb + gradient_size_gb + overhead_gb
    
    # Recomendado (con margen)
    recommended_gb = total_required_gb * 1.3
    
    # Si usamos múltiples GPUs, dividir
    if num_gpus > 1:
        total_required_gb = total_required_gb / num_gpus
        recommended_gb = recommended_gb / num_gpus
    
    return VRAMRequirements(
        model_size_gb=round(model_size_gb, 2),
        lora_size_gb=round(lora_size_gb, 2),
        total_required_gb=round(total_required_gb, 2),
        recommended_gb=round(recommended_gb, 2),
        quantization=quant_type,
        batch_size=batch_size,
        sequence_length=sequence_length
    )


def get_recommended_settings(
    available_vram_gb: float,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"
) -> Dict:
    """
    Obtener configuración recomendada basada en VRAM disponible.
    
    Args:
        available_vram_gb: VRAM disponible en GB
        model_name: Nombre del modelo base
        
    Returns:
        Configuración recomendada
    """
    # Obtener info del modelo
    model_info = get_model_info(model_name)
    
    recommendations = {
        "model": model_name,
        "quantization": "nf4",
        "batch_size": 1,
        "sequence_length": 4096,
        "lora_rank": 16,
        "lora_alpha": 32,
        "epochs": 3,
        "notes": []
    }
    
    # Modelos grandes > 30B parameters
    if model_info.parameter_count >= 30e9:
        if available_vram_gb >= 24:
            recommendations["quantization"] = "nf4"
            recommendations["batch_size"] = 1
            recommendations["lora_rank"] = 32
        elif available_vram_gb >= 16:
            recommendations["quantization"] = "int4"
            recommendations["batch_size"] = 1
            recommendations["lora_rank"] = 16
        else:
            recommendations["notes"].append("Insufficient VRAM for this model. Consider a smaller model.")
            
    # Modelos 14B-30B parameters
    elif model_info.parameter_count >= 14e9:
        if available_vram_gb >= 16:
            recommendations["quantization"] = "nf4"
            recommendations["batch_size"] = 1
            recommendations["lora_rank"] = 16
        elif available_vram_gb >= 12:
            recommendations["quantization"] = "int4"
            recommendations["batch_size"] = 1
            recommendations["lora_rank"] = 8
        else:
            recommendations["notes"].append("Insufficient VRAM for this model. Consider a smaller model.")
            
    # Modelos 7B-14B parameters
    elif model_info.parameter_count >= 7e9:
        if available_vram_gb >= 12:
            recommendations["quantization"] = "nf4"
            recommendations["batch_size"] = 2
            recommendations["lora_rank"] = 16
        elif available_vram_gb >= 8:
            recommendations["quantization"] = "int4"
            recommendations["batch_size"] = 1
            recommendations["lora_rank"] = 8
        else:
            recommendations["notes"].append("Consider using 4-bit quantization.")
            
    # Modelos < 7B parameters
    else:
        if available_vram_gb >= 8:
            recommendations["quantization"] = "nf4"
            recommendations["batch_size"] = 2
            recommendations["lora_rank"] = 16
        elif available_vram_gb >= 6:
            recommendations["quantization"] = "int8"
            recommendations["batch_size"] = 1
            recommendations["lora_rank"] = 8
        else:
            recommendations["notes"].append("Consider using a smaller model.")
    
    # Ajustar sequence_length
    vram_after_model = available_vram_gb - recommendations["batch_size"] * 2
    if vram_after_model > 12:
        recommendations["sequence_length"] = 4096
    elif vram_after_model > 8:
        recommendations["sequence_length"] = 2048
    else:
        recommendations["sequence_length"] = 1024
    
    return recommendations
