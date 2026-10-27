#!/usr/bin/env python3
"""
Script de ingestión de datos para LLM-FineTuner.

Procesa archivos .jsonl crudos y los prepara para entrenamiento.
Formato de entrada: {"text": "..."}
Formato de salida: Dataset listo para entrenamiento con tokenización.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional

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


def read_jsonl_files(input_dir: str, min_length: int = 10, max_length: int = 4096) -> List[Dict]:
    """
    Leer todos los archivos .jsonl del directorio de entrada.
    
    Args:
        input_dir: Directorio con archivos .jsonl
        min_length: Longitud mínima del texto
        max_length: Longitud máxima del texto
        
    Returns:
        Lista de diccionarios con los datos
    """
    input_path = Path(input_dir)
    if not input_path.exists():
        logger.error(f"Input directory not found: {input_dir}")
        raise FileNotFoundError(f"Input directory not found: {input_dir}")
    
    data = []
    jsonl_files = list(input_path.glob("*.jsonl"))
    
    if not jsonl_files:
        logger.warning(f"No .jsonl files found in {input_dir}")
        return data
    
    logger.info(f"Found {len(jsonl_files)} .jsonl file(s) in {input_dir}")
    
    for jsonl_file in jsonl_files:
        logger.info(f"Processing {jsonl_file.name}...")
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    item = json.loads(line)
                    
                    # Validar formato
                    if "text" not in item:
                        logger.warning(f"Line {line_num} in {jsonl_file.name}: missing 'text' field")
                        continue
                    
                    text = item["text"]
                    
                    # Validar longitud
                    text_length = len(text)
                    if text_length < min_length:
                        logger.warning(f"Line {line_num}: text too short ({text_length} < {min_length})")
                        continue
                    if text_length > max_length:
                        logger.warning(f"Line {line_num}: text too long ({text_length} > {max_length})")
                        continue
                    
                    data.append({
                        "text": text,
                        "source": str(jsonl_file.name),
                        "line": line_num,
                        "length": text_length
                    })
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Error parsing JSON on line {line_num} in {jsonl_file.name}: {e}")
                    continue
    
    logger.info(f"Loaded {len(data)} valid entries from {len(jsonl_files)} file(s)")
    return data


def filter_and_clean_data(data: List[Dict], config: Dict) -> List[Dict]:
    """
    Filtrar y limpiar los datos según la configuración.
    
    Args:
        data: Datos crudos
        config: Configuración
        
    Returns:
        Datos filtrados y limpios
    """
    data_config = config.get("data", {})
    min_length = data_config.get("min_length", 10)
    max_length = data_config.get("max_length", 4096)
    
    # Aplicar filtros
    filtered_data = []
    for item in data:
        text = item["text"]
        text_length = len(text)
        
        if min_length <= text_length <= max_length:
            filtered_data.append(item)
    
    logger.info(f"After filtering: {len(filtered_data)} entries ({len(data) - len(filtered_data)} removed)")
    
    return filtered_data


def split_data(data: List[Dict], config: Dict) -> Dict[str, List[Dict]]:
    """
    Dividir datos en train/validation/test según la configuración.
    
    Args:
        data: Datos a dividir
        config: Configuración
        
    Returns:
        Diccionario con splits
    """
    data_config = config.get("data", {})
    val_split = data_config.get("val_split", 0.1)
    test_split = data_config.get("test_split", 0.0)
    
    if val_split <= 0 and test_split <= 0:
        return {"train": data, "validation": [], "test": []}
    
    # Calcular tamaños
    total = len(data)
    val_size = int(total * val_split)
    test_size = int(total * test_split)
    train_size = total - val_size - test_size
    
    # Dividir
    train_data = data[:train_size]
    val_data = data[train_size:train_size + val_size]
    test_data = data[train_size + val_size:]
    
    logger.info(f"Data split: train={len(train_data)}, validation={len(val_data)}, test={len(test_data)}")
    
    return {
        "train": train_data,
        "validation": val_data,
        "test": test_data
    }


def save_processed_data(splits: Dict[str, List[Dict]], output_dir: str) -> None:
    """
    Guardar datos procesados en archivos JSON.
    
    Args:
        splits: Diccionario con splits de datos
        output_dir: Directorio de salida
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_data in splits.items():
        output_file = output_path / f"{split_name}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(split_data, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved {len(split_data)} entries to {output_file}")
    
    # También guardar el dataset completo
    complete_file = output_path / "dataset.json"
    with open(complete_file, 'w', encoding='utf-8') as f:
        json.dump({
            "train": splits["train"],
            "validation": splits["validation"],
            "test": splits["test"]
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved complete dataset to {complete_file}")


def validate_data(data: List[Dict]) -> None:
    """
    Validar que los datos tienen el formato correcto.
    
    Args:
        data: Datos a validar
        
    Raises:
        ValueError: Si hay errores de formato
    """
    errors = []
    
    for i, item in enumerate(data):
        if "text" not in item:
            errors.append(f"Item {i}: missing 'text' field")
        elif not isinstance(item["text"], str):
            errors.append(f"Item {i}: 'text' is not a string")
        elif len(item["text"]) == 0:
            errors.append(f"Item {i}: 'text' is empty")
    
    if errors:
        for error in errors[:10]:  # Mostrar solo primeros 10 errores
            logger.error(error)
        raise ValueError(f"Data validation failed: {len(errors)} errors found")
    
    logger.info("Data validation passed")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Ingestion script for LLM-FineTuner")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to config file")
    parser.add_argument("--input", type=str, default=None, help="Input directory (overrides config)")
    parser.add_argument("--output", type=str, default=None, help="Output directory (overrides config)")
    parser.add_argument("--skip-validation", action="store_true", help="Skip data validation")
    args = parser.parse_args()
    
    try:
        # Cargar configuración
        config = load_config(args.config)
        
        # Override paths si se proporcionan
        input_dir = args.input or config.get("data", {}).get("input_dir", "data/raw")
        output_dir = args.output or config.get("data", {}).get("processed_dir", "data/processed")
        
        logger.info(f"Input directory: {input_dir}")
        logger.info(f"Output directory: {output_dir}")
        
        # Leer datos
        logger.info("Reading .jsonl files...")
        raw_data = read_jsonl_files(
            input_dir=input_dir,
            min_length=config.get("data", {}).get("min_length", 10),
            max_length=config.get("data", {}).get("max_length", 4096)
        )
        
        if not raw_data:
            logger.error("No data loaded. Please check your input files.")
            return
        
        # Filtrar y limpiar
        logger.info("Filtering and cleaning data...")
        filtered_data = filter_and_clean_data(raw_data, config)
        
        if not filtered_data:
            logger.error("No data after filtering. Please check your filters.")
            return
        
        # Validar
        if not args.skip_validation:
            logger.info("Validating data...")
            validate_data(filtered_data)
        
        # Dividir
        logger.info("Splitting data...")
        splits = split_data(filtered_data, config)
        
        # Guardar
        logger.info("Saving processed data...")
        save_processed_data(splits, output_dir)
        
        logger.info("Data ingestion completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during data ingestion: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
