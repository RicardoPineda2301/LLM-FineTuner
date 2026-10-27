#!/usr/bin/env python3
"""
Script de exportación a GGUF para LLM-FineTuner.

Convierte modelos fusionados a formato .gguf para uso portable.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Dict, Optional

import yaml

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_config(config_path: str = "config/config.yaml") -> Dict:
    """Cargar configuración desde YAML."""
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def export_to_gguf(
    model_path: str,
    output_path: str,
    quantization: str = "int4",
    model_type: Optional[str] = None
) -> None:
    """
    Exportar modelo a formato GGUF usando llm-gguf.
    
    Args:
        model_path: Ruta al modelo (directorio o archivo)
        output_path: Ruta de salida para el archivo .gguf
        quantization: Tipo de cuantización (int4, int8, float16, float32)
        model_type: Tipo de modelo (autodetectar si None)
    """
    try:
        # Importar llm-gguf
        from llm_gguf import Model
        
        logger.info(f"Exporting model to GGUF...")
        logger.info(f"Input: {model_path}")
        logger.info(f"Output: {output_path}")
        logger.info(f"Quantization: {quantization}")
        
        # Cargar modelo
        logger.info("Loading model...")
        model = Model.load(model_path, model_type=model_type)
        
        # Configurar cuantización
        quant_map = {
            "int4": "q4_0",
            "int8": "q8_0",
            "float16": "f16",
            "float32": "f32",
            "int4_k_m": "q4_k_m",
            "int4_k": "q4_k",
        }
        
        if quantization not in quant_map:
            logger.warning(f"Unknown quantization '{quantization}', using 'q4_0' as default")
            quant = "q4_0"
        else:
            quant = quant_map[quantization]
        
        # Exportar a GGUF
        logger.info(f"Exporting to GGUF with {quant} quantization...")
        model.export(output_path, quant)
        
        logger.info(f"GGUF model exported successfully to {output_path}")
        
        # Verificar archivo
        output_file = Path(output_path)
        if output_file.exists():
            file_size = output_file.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"File size: {file_size:.2f} MB")
        
    except ImportError:
        logger.error("llm-gguf not installed. Please install with: pip install llm-gguf")
        raise
    except Exception as e:
        logger.error(f"Error exporting to GGUF: {e}", exc_info=True)
        raise


def export_with_llama_cpp(
    model_path: str,
    output_path: str,
    quantization: str = "int4"
) -> None:
    """
    Exportar usando el script de conversión de llama.cpp.
    
    Args:
        model_path: Ruta al modelo
        output_path: Ruta de salida
        quantization: Tipo de cuantización
    """
    import subprocess
    import sys
    
    logger.info("Trying conversion with llama.cpp...")
    
    # Buscar python script de conversión
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        logger.info("Loading model with transformers...")
        
        # Cargar modelo
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        logger.info("Model loaded. Starting conversion...")
        
        # Convertir usando export de transformers
        # Esto puede no funcionar para todos los modelos
        model.save_pretrained(output_path, save_format="gguf")
        
        logger.info(f"Model exported to {output_path}")
        
    except Exception as e:
        logger.error(f"Error with transformers export: {e}")
        raise


def detect_model_type(model_path: str) -> str:
    """
    Detectar tipo de modelo automáticamente.
    
    Args:
        model_path: Ruta al modelo
        
    Returns:
        Tipo de modelo (ej: "qwen2", "mistral", "llama")
    """
    from pathlib import Path
    import json
    
    model_path = Path(model_path)
    
    # Intentar detectar desde config.json
    config_file = model_path / "config.json"
    if config_file.exists():
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        model_name = config.get("model_name", "").lower()
        
        if "qwen" in model_name:
            return "qwen2"
        elif "mistral" in model_name:
            return "mistral"
        elif "llama" in model_name:
            return "llama"
        elif "gemma" in model_name:
            return "gemma"
    
    # Detectar desde estructura de directorios
    if "qwen" in str(model_path).lower():
        return "qwen2"
    elif "mistral" in str(model_path).lower():
        return "mistral"
    elif "llama" in str(model_path).lower():
        return "llama"
    
    return "auto"


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Export script for LLM-FineTuner")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--model", type=str, default=None, help="Model directory or file path")
    parser.add_argument("--output", type=str, default=None, help="Output .gguf file path")
    parser.add_argument("--quant", type=str, default="int4", 
                        help="Quantization type (int4, int8, float16, float32)")
    parser.add_argument("--model-type", type=str, default=None, 
                        help="Model type (qwen2, mistral, llama, gemma, auto)")
    parser.add_argument("--method", type=str, default="llm-gguf",
                        choices=["llm-gguf", "transformers"],
                        help="Export method")
    args = parser.parse_args()
    
    try:
        # Cargar configuración
        config = load_config(args.config)
        
        # Obtener valores
        model_path = args.model
        output_dir = args.output or config.get("export", {}).get("export_dir", "data/output/gguf")
        quantization = args.quant or config.get("export", {}).get("gguf_quantization", "int4")
        model_type = args.model_type
        method = args.method
        
        if not model_path:
            logger.error("Model path is required. Use --model or set in config.")
            return
        
        if not os.path.exists(model_path):
            logger.error(f"Model path not found: {model_path}")
            return
        
        # Crear directorio de salida
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Generar nombre de archivo
        model_name = Path(model_path).name.replace("/", "-")
        output_file = output_path / f"{model_name}-{quantization}.gguf"
        
        # Detectar tipo de modelo si no se especifica
        if model_type is None:
            model_type = detect_model_type(model_path)
            logger.info(f"Detected model type: {model_type}")
        
        # Exportar
        if method == "llm-gguf":
            export_to_gguf(
                model_path=model_path,
                output_path=str(output_file),
                quantization=quantization,
                model_type=model_type
            )
        else:
            export_with_llama_cpp(
                model_path=model_path,
                output_path=str(output_file),
                quantization=quantization
            )
        
        logger.info(f"Export completed! GGUF file saved to {output_file}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
