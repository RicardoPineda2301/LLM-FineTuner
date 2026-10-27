#!/usr/bin/env python3
"""
Script de entrenamiento QLoRA para LLM-FineTuner.

Usa Unsloth para aplicar QLoRA (Quantized Low-Rank Adapters) y entrenar
eficientemente con bajo consumo de VRAM.
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


def load_dataset(dataset_path: str) -> list:
    """
    Cargar dataset desde archivo JSON.
    
    Args:
        dataset_path: Ruta al archivo JSON con los datos
        
    Returns:
        Lista de textos para entrenamiento
    """
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        logger.error(f"Dataset file not found: {dataset_path}")
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    
    import json
    with open(dataset_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extraer textos (asumimos formato {"train": [...], "validation": [...]})
    texts = []
    if isinstance(data, dict):
        if "train" in data:
            texts = [item["text"] for item in data["train"]]
        else:
            texts = [item["text"] for item in data if "text" in item]
    elif isinstance(data, list):
        texts = [item["text"] for item in data if "text" in item]
    
    logger.info(f"Loaded {len(texts)} training examples from {dataset_path}")
    return texts


def setup_training_config(config: Dict) -> Dict:
    """
    Configurar parámetros de entrenamiento para Unsloth.
    
    Args:
        config: Configuración cargada
        
    Returns:
        Configuración procesada para Unsloth
    """
    model_config = config.get("model", {})
    training_config = config.get("training", {})
    hardware_config = config.get("hardware", {})
    
    # Parámetros LoRA
    lora_rank = training_config.get("lora_rank", 16)
    lora_alpha = training_config.get("lora_alpha", 32)
    lora_dropout = training_config.get("lora_dropout", 0.05)
    
    # Parámetros de entrenamiento
    epochs = training_config.get("epochs", 3)
    batch_size = training_config.get("batch_size", 2)
    learning_rate = training_config.get("learning_rate", 2e-4)
    max_seq_length = training_config.get("max_seq_length", 4096)
    
    # Optimizador
    optimizer = training_config.get("optimizer", "paged_adamw_8bit")
    weight_decay = training_config.get("weight_decay", 0.01)
    warmup_steps = training_config.get("warmup_steps", 100)
    
    # Gradient accumulation
    gradient_accumulation_steps = training_config.get("gradient_accumulation_steps", 1)
    
    # Cuantización
    quantization = model_config.get("quantization", "nf4")
    
    # Dispositivo
    device = model_config.get("device", "auto")
    
    # GPU
    num_gpus = hardware_config.get("num_gpus", 1)
    
    return {
        "model_name": model_config.get("base", "Qwen/Qwen2.5-7B-Instruct"),
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "lora_dropout": lora_dropout,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "max_seq_length": max_seq_length,
        "optimizer": optimizer,
        "weight_decay": weight_decay,
        "warmup_steps": warmup_steps,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "quantization": quantization,
        "device": device,
        "num_gpus": num_gpus,
        "fp16": training_config.get("fp16", True),
        "bf16": training_config.get("bf16", False),
        "tf32": training_config.get("tf32", False),
    }


def train_with_unsloth(
    model_name: str,
    dataset: list,
    output_dir: str,
    training_config: Dict,
    validation_data: Optional[list] = None
) -> None:
    """
    Entrenar modelo usando Unsloth con QLoRA.
    
    Args:
        model_name: Nombre del modelo base en HuggingFace
        dataset: Lista de textos para entrenamiento
        output_dir: Directorio de salida para el adaptador
        training_config: Configuración de entrenamiento
        validation_data: Datos de validación (opcional)
    """
    from unsloth import FastLanguageModel
    import torch
    
    logger.info(f"Starting training with Unsloth...")
    logger.info(f"Model: {model_name}")
    logger.info(f"Dataset size: {len(dataset)}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Training config: {training_config}")
    
    # Crear directorio de salida
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Inicializar modelo con QLoRA
    logger.info("Initializing model with QLoRA...")
    
    try:
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=training_config["max_seq_length"],
            dtype=torch.float16 if training_config["fp16"] else None,
            load_in_4bit=training_config["quantization"] in ["nf4", "fp4"],
            load_in_8bit=training_config["quantization"] in ["int8"],
            device_map=training_config["device"],
            token=os.getenv("HUGGINGFACE_TOKEN", None)
        )
        
        logger.info("Model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise
    
    # Aplicar LoRA
    logger.info("Applying LoRA...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=training_config["lora_rank"],
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=training_config["lora_alpha"],
        lora_dropout=training_config["lora_dropout"],
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )
    
    logger.info("LoRA applied successfully")
    
    # Tokenizar dataset
    logger.info("Tokenizing dataset...")
    
    def formatting_prompts_func(examples):
        """Formatear ejemplos para entrenamiento."""
        texts = []
        for example in examples:
            # Asumimos que example es el texto directamente
            if isinstance(example, dict):
                text = example.get("text", "")
            else:
                text = str(example)
            texts.append(text)
        return texts
    
    # Convertir dataset a formato para Unsloth
    # Unsloth espera una lista de strings
    train_dataset = dataset
    
    # Validación
    if validation_data:
        eval_dataset = validation_data
    else:
        eval_dataset = None
    
    # Entrenar
    logger.info("Starting training...")
    
    trainer = FastLanguageModel.SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        formatting_prompts_func=formatting_prompts_func,
        max_seq_length=training_config["max_seq_length"],
        num_train_epochs=training_config["epochs"],
        per_device_train_batch_size=training_config["batch_size"],
        gradient_accumulation_steps=training_config["gradient_accumulation_steps"],
        learning_rate=training_config["learning_rate"],
        weight_decay=training_config["weight_decay"],
        warmup_steps=training_config["warmup_steps"],
        lr_scheduler_type="cosine",
        logging_steps=training_config.get("logging_steps", 10),
        save_steps=training_config.get("save_steps", 100),
        eval_steps=training_config.get("eval_steps", 100),
        output_dir=str(output_path),
        overwrite_output_dir=True,
        # Parámetros adicionales
        fp16=training_config["fp16"],
        bf16=training_config["bf16"],
        tf32=training_config["tf32"],
        # Early stopping
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        early_stopping_patience=training_config.get("early_stopping_patience", 3),
    )
    
    logger.info("Trainer initialized. Starting training loop...")
    
    try:
        # Iniciar entrenamiento
        trainer_stats = trainer.train()
        
        logger.info("Training completed!")
        logger.info(f"Training stats: {trainer_stats}")
        
        # Guardar adaptador
        adapter_path = output_path / "adapter_model.safetensors"
        logger.info(f"Saving adapter to {adapter_path}...")
        
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        # Guardar configuración del adaptador
        adapter_config = {
            "model_name": model_name,
            "lora_rank": training_config["lora_rank"],
            "lora_alpha": training_config["lora_alpha"],
            "lora_dropout": training_config["lora_dropout"],
            "training_epochs": training_config["epochs"],
            "batch_size": training_config["batch_size"],
            "learning_rate": training_config["learning_rate"],
            "max_seq_length": training_config["max_seq_length"],
            "quantization": training_config["quantization"],
            "base_model": model_name,
            "dataset_size": len(dataset),
        }
        
        config_file = output_path / "adapter_config.json"
        import json
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(adapter_config, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Adapter saved to {output_path}")
        logger.info(f"Adapter config saved to {config_file}")
        
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        # Guardar checkpoint
        checkpoint_dir = output_path / "checkpoint"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving checkpoint to {checkpoint_dir}...")
        trainer.save_model(str(checkpoint_dir))
        logger.info("Checkpoint saved. Training can be resumed.")
        raise
    except Exception as e:
        logger.error(f"Error during training: {e}", exc_info=True)
        raise
    finally:
        # Limpiar GPU
        import gc
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        logger.info("GPU cache cleared")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="QLoRA training script for LLM-FineTuner")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--dataset", type=str, default=None, help="Path to training dataset (JSON)")
    parser.add_argument("--validation", type=str, default=None, help="Path to validation dataset (JSON)")
    parser.add_argument("--model", type=str, default=None, help="Model name (overrides config)")
    parser.add_argument("--output", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides config)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--learning-rate", type=float, default=None, help="Learning rate (overrides config)")
    args = parser.parse_args()
    
    try:
        # Cargar configuración
        config = load_config(args.config)
        
        # Override desde argumentos
        model_name = args.model or config.get("model", {}).get("base", "Qwen/Qwen2.5-7B-Instruct")
        output_dir = args.output or config.get("model", {}).get("output_dir", "data/output/adapters")
        dataset_path = args.dataset or (Path(config.get("data", {}).get("processed_dir", "data/processed")) / "train.json")
        validation_path = args.validation or (Path(config.get("data", {}).get("processed_dir", "data/processed")) / "validation.json")
        
        # Cargar dataset
        logger.info(f"Loading dataset from {dataset_path}...")
        dataset = load_dataset(str(dataset_path))
        
        # Cargar validación si existe
        validation_data = None
        if validation_path.exists():
            logger.info(f"Loading validation data from {validation_path}...")
            validation_data = load_dataset(str(validation_path))
        
        if not dataset:
            logger.error("No training data loaded. Please check your dataset.")
            return
        
        # Procesar configuración de entrenamiento
        training_config = setup_training_config(config)
        
        # Override desde argumentos
        if args.epochs is not None:
            training_config["epochs"] = args.epochs
        if args.batch_size is not None:
            training_config["batch_size"] = args.batch_size
        if args.learning_rate is not None:
            training_config["learning_rate"] = args.learning_rate
        training_config["model_name"] = model_name
        
        # Crear directorio de salida
        timestamp = __import__('datetime').datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(output_dir) / f"{model_name.replace('/', '_')}-lora-{timestamp}"
        
        # Entrenar
        train_with_unsloth(
            model_name=model_name,
            dataset=dataset,
            output_dir=str(output_dir),
            training_config=training_config,
            validation_data=validation_data
        )
        
        logger.info(f"Training completed! Adapter saved to {output_dir}")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
