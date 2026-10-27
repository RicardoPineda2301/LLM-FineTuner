"""
Cargador de datasets para LLM-FineTuner.

Soporta múltiples formatos de datos y proporciona utilidades para preprocesamiento.
"""

import json
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Iterator

logger = logging.getLogger(__name__)


class DatasetFormat(Enum):
    """Formatos de dataset soportados."""
    JSONL = "jsonl"
    JSON = "json"
    CSV = "csv"
    TXT = "txt"


@dataclass
class DatasetExample:
    """Ejemplo de dataset."""
    text: str
    source: Optional[str] = None
    metadata: Optional[Dict] = None
    
    def to_dict(self) -> Dict:
        """Convertir a diccionario."""
        result = {"text": self.text}
        if self.source:
            result["source"] = self.source
        if self.metadata:
            result["metadata"] = self.metadata
        return result


@dataclass
class Dataset:
    """Dataset completo."""
    examples: List[DatasetExample]
    format: DatasetFormat
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __iter__(self) -> Iterator[DatasetExample]:
        return iter(self.examples)
    
    def split(self, ratios: Dict[str, float]) -> Dict[str, "Dataset"]:
        """
        Dividir dataset en múltiples splits.
        
        Args:
            ratios: Diccionario con ratios para cada split (ej: {"train": 0.8, "val": 0.2})
            
        Returns:
            Diccionario con datasets divididos
        """
        from random import shuffle
        import math
        
        total = len(self.examples)
        shuffled = self.examples.copy()
        shuffle(shuffled)
        
        splits = {}
        start = 0
        
        for split_name, ratio in ratios.items():
            size = math.floor(total * ratio)
            splits[split_name] = Dataset(
                examples=shuffled[start:start + size],
                format=self.format
            )
            start += size
        
        return splits
    
    def filter(self, min_length: int = 0, max_length: int = float('inf')) -> "Dataset":
        """
        Filtrar ejemplos por longitud.
        
        Args:
            min_length: Longitud mínima
            max_length: Longitud máxima
            
        Returns:
            Dataset filtrado
        """
        filtered = [
            ex for ex in self.examples 
            if min_length <= len(ex.text) <= max_length
        ]
        return Dataset(examples=filtered, format=self.format)
    
    def to_json(self, path: str) -> None:
        """
        Guardar dataset como JSON.
        
        Args:
            path: Ruta al archivo de salida
        """
        data = [ex.to_dict() for ex in self.examples]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def to_jsonl(self, path: str) -> None:
        """
        Guardar dataset como JSONL.
        
        Args:
            path: Ruta al archivo de salida
        """
        with open(path, 'w', encoding='utf-8') as f:
            for ex in self.examples:
                f.write(json.dumps(ex.to_dict(), ensure_ascii=False) + '\n')
    
    @classmethod
    def from_list(cls, texts: List[str], format: DatasetFormat = DatasetFormat.JSONL) -> "Dataset":
        """
        Crear dataset desde lista de textos.
        
        Args:
            texts: Lista de textos
            format: Formato del dataset
            
        Returns:
            Dataset
        """
        examples = [DatasetExample(text=text) for text in texts]
        return cls(examples=examples, format=format)


def load_dataset(
    path: Union[str, Path],
    format: Optional[DatasetFormat] = None,
    text_field: str = "text"
) -> Dataset:
    """
    Cargar dataset desde archivo.
    
    Args:
        path: Ruta al archivo
        format: Formato (autodetectar si None)
        text_field: Campo que contiene el texto (para JSON)
        
    Returns:
        Dataset cargado
        
    Raises:
        FileNotFoundError: Si el archivo no existe
        ValueError: Si el formato no es válido
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    
    # Autodetectar formato
    if format is None:
        suffix = path.suffix.lower()
        if suffix == ".jsonl":
            format = DatasetFormat.JSONL
        elif suffix == ".json":
            format = DatasetFormat.JSON
        elif suffix == ".csv":
            format = DatasetFormat.CSV
        elif suffix == ".txt":
            format = DatasetFormat.TXT
        else:
            raise ValueError(f"Unknown dataset format for extension {suffix}")
    
    # Cargar según formato
    if format == DatasetFormat.JSONL:
        return _load_jsonl(path, text_field)
    elif format == DatasetFormat.JSON:
        return _load_json(path, text_field)
    elif format == DatasetFormat.CSV:
        return _load_csv(path, text_field)
    elif format == DatasetFormat.TXT:
        return _load_txt(path)
    else:
        raise ValueError(f"Unsupported format: {format}")


def _load_jsonl(path: Path, text_field: str = "text") -> Dataset:
    """Cargar dataset desde JSONL."""
    examples = []
    
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                text = data.get(text_field)
                if not text:
                    logger.warning(f"Line {line_num}: missing text field '{text_field}'")
                    continue
                
                examples.append(DatasetExample(
                    text=str(text),
                    source=str(path),
                    metadata={k: v for k, v in data.items() if k != text_field}
                ))
            except json.JSONDecodeError as e:
                logger.error(f"Error parsing JSONL line {line_num}: {e}")
                continue
    
    logger.info(f"Loaded {len(examples)} examples from {path}")
    return Dataset(examples=examples, format=DatasetFormat.JSONL)


def _load_json(path: Path, text_field: str = "text") -> Dataset:
    """Cargar dataset desde JSON."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    examples = []
    
    if isinstance(data, list):
        for item in data:
            text = item.get(text_field)
            if not text:
                continue
            examples.append(DatasetExample(
                text=str(text),
                source=str(path),
                metadata={k: v for k, v in item.items() if k != text_field}
            ))
    elif isinstance(data, dict):
        # Asumir estructura {"train": [...], "validation": [...]}
        for split_name, split_data in data.items():
            if isinstance(split_data, list):
                for item in split_data:
                    text = item.get(text_field)
                    if not text:
                        continue
                    metadata = {k: v for k, v in item.items() if k != text_field}
                    metadata["_split"] = split_name
                    examples.append(DatasetExample(
                        text=str(text),
                        source=str(path),
                        metadata=metadata
                    ))
    
    logger.info(f"Loaded {len(examples)} examples from {path}")
    return Dataset(examples=examples, format=DatasetFormat.JSON)


def _load_csv(path: Path, text_field: str = "text") -> Dataset:
    """Cargar dataset desde CSV."""
    try:
        import csv
    except ImportError:
        raise ImportError("csv module is required for CSV format")
    
    examples = []
    
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row.get(text_field)
            if not text:
                continue
            examples.append(DatasetExample(
                text=str(text),
                source=str(path),
                metadata={k: v for k, v in row.items() if k != text_field}
            ))
    
    logger.info(f"Loaded {len(examples)} examples from {path}")
    return Dataset(examples=examples, format=DatasetFormat.CSV)


def _load_txt(path: Path) -> Dataset:
    """Cargar dataset desde TXT."""
    examples = []
    
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
        
        # Dividir por líneas vacías o saltos de línea
        parts = [part.strip() for part in content.split('\n\n') if part.strip()]
        
        for part in parts:
            examples.append(DatasetExample(
                text=part,
                source=str(path)
            ))
    
    logger.info(f"Loaded {len(examples)} examples from {path}")
    return Dataset(examples=examples, format=DatasetFormat.TXT)


def load_multiple_datasets(
    paths: List[Union[str, Path]],
    text_field: str = "text"
) -> Dataset:
    """
    Cargar múltiples datasets y combinarlos.
    
    Args:
        paths: Lista de rutas a archivos
        text_field: Campo de texto
        
    Returns:
        Dataset combinado
    """
    all_examples = []
    
    for path in paths:
        dataset = load_dataset(path, text_field=text_field)
        all_examples.extend(dataset.examples)
    
    return Dataset(examples=all_examples, format=DatasetFormat.JSONL)


def create_train_val_test_split(
    dataset: Dataset,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1
) -> Dict[str, Dataset]:
    """
    Dividir dataset en train/validation/test.
    
    Args:
        dataset: Dataset a dividir
        val_ratio: Proporción para validación
        test_ratio: Proporción para test
        
    Returns:
        Diccionario con los splits
    """
    from random import shuffle
    
    total = len(dataset.examples)
    shuffled = dataset.examples.copy()
    shuffle(shuffled)
    
    val_size = int(total * val_ratio)
    test_size = int(total * test_ratio)
    train_size = total - val_size - test_size
    
    return {
        "train": Dataset(
            examples=shuffled[:train_size],
            format=dataset.format
        ),
        "validation": Dataset(
            examples=shuffled[train_size:train_size + val_size],
            format=dataset.format
        ),
        "test": Dataset(
            examples=shuffled[train_size + val_size:],
            format=dataset.format
        )
    }


def preprocess_text(text: str) -> str:
    """
    Preprocesar texto para entrenamiento.
    
    Args:
        text: Texto crudo
        
    Returns:
        Texto preprocesado
    """
    # Eliminar espacios múltiples
    text = ' '.join(text.split())
    
    # Asegurar que termine con salto de línea
    if not text.endswith('\n'):
        text += '\n'
    
    return text


def validate_dataset_format(dataset: Dataset, check_instructions: bool = True) -> List[str]:
    """
    Validar formato del dataset.
    
    Args:
        dataset: Dataset a validar
        check_instructions: Verificar que los ejemplos tienen formato de instrucciones
        
    Returns:
        Lista de errores
    """
    errors = []
    
    for i, example in enumerate(dataset.examples):
        text = example.text.strip()
        
        if not text:
            errors.append(f"Example {i}: empty text")
            continue
        
        if check_instructions:
            # Verificar formato [INST] ... [/INST]
            if "[INST]" not in text or "[/INST]" not in text:
                errors.append(f"Example {i}: missing [INST] and [/INST] tags")
        
        if len(text) > 4096:
            errors.append(f"Example {i}: text too long ({len(text)} > 4096)")
    
    return errors
