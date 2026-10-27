#!/usr/bin/env python3
"""
Script de fusión para LLM-FineTuner.

Combina el modelo base con el adaptador LoRA entrenado para crear
a un modelo fusionado completo.
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


def merge_model_and_adapter(
    base_model_name: str,
    adapter_path: str,
    output_dir: str,
    quantization: Optional[str] = None,
    device: str = "auto"
) -> None:
    """
    Fusionar modelo base con adaptador LoRA.
    
    Args:
        base_model_name: Nombre del modelo base en HuggingFace
        adapter_path: Ruta al directorio del adaptador
        output_dir: Directorio de salida para el modelo fusionado
        quantization: Cuantización para el modelo final (opcional)
        device: Dispositivo a usar (cuda, cpu, auto)
    """
    from unsloth import FastLanguageModel
    import torch
    
    logger.info(f"Merging model and adapter...")
    logger.info(f"Base model: {base_model_name}")
    logger.info(f"Adapter: {adapter_path}")
    logger.info(f"Output: {output_dir}")
    
    # Crear directorios
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Cargar modelo base
    logger.info("Loading base model...")
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model_name,
            max_seq_length=4096,  # Longitud máxima por defecto
            dtype=torch.float16,
            load_in_4bit=False,  # Cargar en full precision para fusión
            device_map=device,
            token=os.getenv("HUGGINGFACE_TOKEN", None)
        )
        
        logger.info("Base model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading base model: {e}")
        raise
    
    # Cargar adaptador
    logger.info("Loading adapter...")
    adapter_path = Path(adapter_path)
    
    try:
        model = FastLanguageModel.from_pretrained(
            model_name=str(adapter_path),
            model=model,
            max_seq_length=4096,
        )
        
        logger.info("Adapter loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading adapter: {e}")
        raise
    
    # Fusionar modelo + adaptador
    logger.info("Merging model and adapter...")
    
    try:
        # Fusionar (esto aplicará los pesos LoRA al modelo base)
        merged_model = model.merge_and_unload()
        
        logger.info("Model and adapter merged successfully")
        
    except Exception as e:
        logger.error(f"Error merging model and adapter: {e}")
        raise
    
    # Guardar modelo fusionado
    logger.info("Saving merged model...")
    
    try:
        merged_model.save_pretrained(str(output_path))
        tokenizer.save_pretrained(str(output_path))
        
        logger.info(f"Merged model saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving merged model: {e}")
        raise
    
    # Opcional: cuantizar el modelo fusionado
    if quantization:
        logger.info(f"Quantizing merged model to {quantization}...")
        quantize_merged_model(output_path, quantization)
    
    # Guardar configuración
    merged_config = {
        "base_model": base_model_name,
        "adapter": str(adapter_path),
        "output_dir": str(output_path),
        "quantization": quantization,
        "merged_at": __import__('datetime').datetime.now().isoformat()
    }
    
    config_file = output_path / "merged_config.json"
    import json
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump(merged_config, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Merge config saved to {config_file}")
    
    # Limpiar GPU
    import gc
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("GPU cache cleared")


def quantize_merged_model(model_path: Path, quantization: str) -> None:
    """
    Cuantizar modelo fusionado.
    
    Args:
        model_path: Ruta al modelo fusionado
        quantization: Tipo de cuantización (int8, int4, nf4, fp4)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    import torch
    
    logger.info(f"Quantizing model to {quantization}...")
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True if quantization in ["nf4", "fp4"] else False,
        load_in_8bit=True if quantization == "int8" else False,
        bnb_4bit_quant_type=quantization if quantization in ["nf4", "fp4"] else "nf4",
        bnb_4bit_use_double_quant=False,
        bnb_8bit_quant_type="int8" if quantization == "int8" else "int8",
    )
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path),
            quantization_config=quant_config if quantization in ["nf4", "fp4", "int8"] else None,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        
        # Guardar modelo cuantizado
        quant_path = model_path.parent / f"{model_path.name}-{quantization}"
        model.save_pretrained(str(quant_path))
        
        logger.info(f"Quantized model saved to {quant_path}")
        
    except Exception as e:
        logger.error(f"Error quantizing model: {e}")
        raise


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Merge script for LLM-FineTuner")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--base", type=str, default=None, help="Base model name (overrides config)")
    parser.add_argument("--adapter", type=str, default=None, help="Adapter directory path")
    parser.add_argument("--output", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--quantize", type=str, default=None, 
                        help="Quantization for merged model (nf4, fp4, int8, int4)")
    parser.add_argument("--device", type=str, default="auto", 
                        help="Device to use (cuda, cpu, auto)")
    args = parser.parse_args()
    
    try:
        # Cargar configuración
        config = load_config(args.config)
        
        # Obtener valores
        base_model = args.base or config.get("model", {}).get("base", "Qwen/Qwen2.5-7B-Instruct")
        adapter_path = args.adapter
        output_dir = args.output or config.get("merge", {}).get("merged_dir", "data/output/merged")
        quantization = args.quantize or config.get("model", {}).get("quantization", None)
        device = args.device or config.get("model", {}).get("device", "auto")
        
        if not adapter_path:
            logger.error("Adapter path is required. Use --adapter or set in config.")
            return
        
        if not os.path.exists(adapter_path):
            logger.error(f"Adapter directory not found: {adapter_path}")
            return
        
        # Fusionar
        merge_model_and_adapter(
            base_model_name=base_model,
            adapter_path=adapter_path,
            output_dir=output_dir,
            quantization=quantization,
            device=device
        )
        
        logger.info("Merge completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
