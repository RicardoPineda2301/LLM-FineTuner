#!/usr/bin/env python3
"""
Script para crear datasets de fine-tuning desde código fuente.

Extrae código, comentarios y documentación de repositorios para crear
datasets en formato .jsonl para fine-tuning.
"""

import argparse
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Set

logger = logging.getLogger(__name__)

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


class DatasetCreator:
    """Creador de datasets para fine-tuning."""
    
    def __init__(
        self,
        input_path: str,
        output_path: str,
        include_extensions: Set[str] = None,
        exclude_extensions: Set[str] = None,
        include_dirs: Set[str] = None,
        exclude_dirs: Set[str] = None,
        max_file_size: int = 1024 * 1024,  # 1MB
        max_examples_per_file: int = 50,
        min_line_length: int = 10,
        max_line_length: int = 2000
    ):
        """
        Inicializar el creador de datasets.
        
        Args:
            input_path: Ruta de entrada (archivo o directorio)
            output_path: Ruta de salida para el dataset
            include_extensions: Extensiones a incluir
            exclude_extensions: Extensiones a excluir
            include_dirs: Directorios a incluir
            exclude_dirs: Directorios a excluir
            max_file_size: Tamaño máximo de archivo a procesar
            max_examples_per_file: Máximo de ejemplos por archivo
            min_line_length: Longitud mínima de línea
            max_line_length: Longitud máxima de línea
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path)
        
        # Extensiones por defecto
        if include_extensions is None:
            include_extensions = {
                '.py', '.java', '.js', '.ts', '.go', '.rust', '.cpp', '.c', '.h',
                '.hpp', '.php', '.rb', '.swift', '.kt', '.scala', '.md',
                '.txt', '.rst', '.yml', '.yaml', '.json', '.xml', '.html', '.css'
            }
        
        if exclude_extensions is None:
            exclude_extensions = {
                '.min.js', '.min.css', '.bin', '.exe', '.dll', '.so',
                '.png', '.jpg', '.jpeg', '.gif', '.svg', '.ico',
                '.pdf', '.docx', '.xlsx', '.pptx', '.zip', '.tar', '.gz',
                '.jar', '.war', '.ear', '.class', '.o', '.a', '.lib',
                '.log', '.lock', '.bak', '.tmp', '.temp', '.swp'
            }
        
        self.include_extensions = include_extensions
        self.exclude_extensions = exclude_extensions
        self.include_dirs = include_dirs or {'src', 'app', 'lib', 'server', 'client', 'api'}
        self.exclude_dirs = exclude_dirs or {
            'node_modules', '.git', '.svn', '__pycache__', '.venv', 'venv',
            'dist', 'build', 'target', 'bin', 'obj', 'temp', 'tmp',
            'logs', 'log', 'coverage', '.pytest_cache', '.mypy_cache'
        }
        
        self.max_file_size = max_file_size
        self.max_examples_per_file = max_examples_per_file
        self.min_line_length = min_line_length
        self.max_line_length = max_line_length
        
        self.examples = []
    
    def should_include_file(self, file_path: Path) -> bool:
        """Determinar si un archivo debe incluirse."""
        # Excluir por extensión
        if any(file_path.name.lower().endswith(ext.lower()) for ext in self.exclude_extensions):
            return False
        
        # Incluir por extensión
        if not any(file_path.name.lower().endswith(ext.lower()) for ext in self.include_extensions):
            return False
        
        # Excluir por directorio
        path_parts = file_path.parts
        for excluded_dir in self.exclude_dirs:
            if excluded_dir in [p.lower() for p in path_parts]:
                return False
        
        # Incluir por directorio (si se especifica)
        if self.include_dirs:
            path_parts_lower = [p.lower() for p in path_parts]
            if not any(dir in path_parts_lower for dir in self.include_dirs):
                return False
        
        # Excluir por tamaño
        if file_path.stat().st_size > self.max_file_size:
            return False
        
        return True
    
    def extract_examples_from_file(self, file_path: Path) -> List[Dict]:
        """Extraer ejemplos de un archivo."""
        examples = []
        
        try:
            # Leer el archivo según su extensión
            extension = file_path.suffix.lower()
            content = self.read_file(file_path, extension)
            
            if not content:
                return examples
            
            # Procesar según tipo de archivo
            if extension in {'.py', '.js', '.ts', '.java', '.go', '.rust', 
                            '.cpp', '.c', '.h', '.hpp', '.php', '.rb', 
                            '.swift', '.kt', '.scala'}:
                examples = self.extract_from_code(file_path, content, extension)
            elif extension in {'.md', '.txt', '.rst'}:
                examples = self.extract_from_text(file_path, content)
            elif extension in {'.json', '.yaml', '.yml'}:
                examples = self.extract_from_config(file_path, content)
            elif extension in {'.html', '.xml', '.css'}:
                examples = self.extract_from_markup(file_path, content)
        
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
        
        return examples
    
    def read_file(self, file_path: Path, extension: str) -> str:
        """Leer contenido de archivo."""
        try:
            with open(file_path, 'r', encoding=self.get_encoding(extension)) as f:
                return f.read()
        except UnicodeDecodeError:
            # Intentar con UTF-8 con BOM
            try:
                with open(file_path, 'r', encoding='utf-8-sig') as f:
                    return f.read()
            except UnicodeDecodeError:
                # Intentar con Latin-1
                with open(file_path, 'r', encoding='latin-1') as f:
                    return f.read()
    
    def get_encoding(self, extension: str) -> str:
        """Obtener encoding apropiado según extensión."""
        if extension in {'.md', '.txt', '.rst', '.py', '.js', '.ts', '.json', 
                        '.yaml', '.yml', '.html', '.xml', '.css', '.java',
                        '.c', '.cpp', '.h', '.hpp', '.go', '.rust', '.php',
                        '.rb', '.swift', '.kt', '.scala'}:
            return 'utf-8'
        return 'latin-1'
    
    def extract_from_code(self, file_path: Path, content: str, extension: str) -> List[Dict]:
        """Extraer ejemplos de código fuente."""
        examples = []
        
        # Dividir en líneas
        lines = content.split('\n')
        
        # Extraer funciones/métodos
        function_patterns = {
            '.py': r'(?:def\s+\w+\s*\(.*?\):.*?(?=\n\s*(?:def|class|#|$))',
            '.js': r'(?:function\s+\w+\s*\(.*?\).*?\{(?:[^{}]|\{[^{}]*\})*\})|(?:const\s+\w+\s*=\s*function\s*\(.*?\)\s*\{.*?\})',
            '.java': r'(?:public|private|protected|static|final|\s+\w+\s*\(.*?\)\s*\{.*?\})',
            '.go': r'(?:func\s+\w+\s*\(.*?\)\s+\w+\s*\{.*?\})',
        }
        
        pattern = function_patterns.get(extension, r'(?:def|class|function|func|public|private|protected)\s+\w+')
        
        # Buscar bloques de código
        code_blocks = re.findall(r'([\s\S]{1,500}?)(?=\n(?:def|class|function|func|public|private|protected|#|$))', content)
        
        # Extraer funciones
        functions = re.findall(pattern, content, re.DOTALL)
        
        for i, func in enumerate(functions[:self.max_examples_per_file]):
            func_clean = func.strip()
            if self.min_line_length <= len(func_clean) <= self.max_line_length:
                example = {
                    "text": f"[INST] Explica este código:\n```{extension[1:]}\n{func_clean}\n``` [/INST]\n",
                    "source": str(file_path),
                    "type": "code_explanation"
                }
                examples.append(example)
        
        # Extraer comentarios
        comments = self.extract_comments(content, extension)
        for comment in comments[:self.max_examples_per_file]:
            if self.min_line_length <= len(comment) <= self.max_line_length:
                example = {
                    "text": f"[INST] Documenta este comentario:\n{comment} [/INST]\n",
                    "source": str(file_path),
                    "type": "comment_documentation"
                }
                examples.append(example)
        
        # Extraer líneas individuales (para ejemplos pequeños)
        for i, line in enumerate(lines):
            line = line.strip()
            if self.min_line_length <= len(line) <= self.max_line_length:
                # Saltar líneas vacías y comentarios simples
                if line and not line.startswith(('//', '#', '/*', '*', '---')):
                    example = {
                        "text": f"[INST] Escribe documentación para esta línea de código:\n{line} [/INST]\n",
                        "source": str(file_path),
                        "type": "line_documentation"
                    }
                    examples.append(example)
                    
                    # Limitar ejemplos
                    if len(examples) >= self.max_examples_per_file:
                        break
        
        return examples
    
    def extract_comments(self, content: str, extension: str) -> List[str]:
        """Extraer comentarios del código."""
        comments = []
        
        # Patrones de comentarios por lenguaje
        patterns = {
            '.py': r'(?:#.*?$|\"\"\"[\s\S]*?\"\"\"|\'\'\'[\s\S]*?\'\'\')',
            '.js': r'(?:\/\/[^\n]*|\/\*[\s\S]*?\*\/|\/\/[^\n]*)',
            '.java': r'(?:\/\/[^\n]*|\/\*[\s\S]*?\*\/|\/\/[^\n]*)',
            '.go': r'(?:\/\/[^\n]*|\/\*[\s\S]*?\*\/|\/\/[^\n]*)',
            '.c': r'(?:\/\/[^\n]*|\/\*[\s\S]*?\*\/|\/\/[^\n]*)',
            '.cpp': r'(?:\/\/[^\n]*|\/\*[\s\S]*?\*\/|\/\/[^\n]*)',
            '.rust': r'(?:\/\/[^\n]*|\/\*[\s\S]*?\*\/|\/\/[^\n]*)',
        }
        
        pattern = patterns.get(extension, r'(?:#.*?$|\/\/[^\n]*|\/\*[\s\S]*?\*\/)')
        
        matches = re.findall(pattern, content, re.MULTILINE)
        for match in matches:
            comment = match.strip()
            if len(comment) >= self.min_line_length:
                comments.append(comment)
        
        return comments
    
    def extract_from_text(self, file_path: Path, content: str) -> List[Dict]:
        """Extraer ejemplos de archivos de texto."""
        examples = []
        
        # Dividir en párrafos
        paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
        
        for para in paragraphs[:self.max_examples_per_file]:
            if self.min_line_length <= len(para) <= self.max_line_length:
                example = {
                    "text": f"[INST] Genera una pregunta basada en este texto:\n{para} [/INST]\n",
                    "source": str(file_path),
                    "type": "text_qa"
                }
                examples.append(example)
        
        # Extraer líneas individuales
        lines = content.split('\n')
        for line in lines:
            line = line.strip()
            if self.min_line_length <= len(line) <= self.max_line_length:
                example = {
                    "text": f"[INST] Resume esta línea:\n{line} [/INST]\n",
                    "source": str(file_path),
                    "type": "line_summary"
                }
                examples.append(example)
                
                if len(examples) >= self.max_examples_per_file:
                    break
        
        return examples
    
    def extract_from_config(self, file_path: Path, content: str) -> List[Dict]:
        """Extraer ejemplos de archivos de configuración."""
        examples = []
        
        # Extraer claves y valores
        if file_path.suffix.lower() == '.json':
            try:
                data = json.loads(content)
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        text = f"{key}: {json.dumps(value, indent=2)}"
                    else:
                        text = f"{key}: {value}"
                    
                    if self.min_line_length <= len(text) <= self.max_line_length:
                        example = {
                            "text": f"[INST] Explica esta configuración:\n{text} [/INST]\n",
                            "source": str(file_path),
                            "type": "config_explanation"
                        }
                        examples.append(example)
                        
                        if len(examples) >= self.max_examples_per_file:
                            break
            except json.JSONDecodeError:
                pass
        else:
            # YAML
            lines = content.split('\n')
            for line in lines:
                line = line.strip()
                if line and not line.startswith('#') and self.min_line_length <= len(line) <= self.max_line_length:
                    example = {
                        "text": f"[INST] Explica esta línea de configuración:\n{line} [/INST]\n",
                        "source": str(file_path),
                        "type": "config_explanation"
                    }
                    examples.append(example)
                    
                    if len(examples) >= self.max_examples_per_file:
                        break
        
        return examples
    
    def extract_from_markup(self, file_path: Path, content: str) -> List[Dict]:
        """Extraer ejemplos de archivos HTML/XML."""
        examples = []
        
        # Extraer texto (quitar tags)
        if file_path.suffix.lower() == '.html':
            text_content = re.sub(r'<[^>]+>', '', content)
        else:
            text_content = re.sub(r'<[^>]+>', '', content)
        
        # Dividir en párrafos
        paragraphs = [p.strip() for p in text_content.split('\n\n') if p.strip()]
        
        for para in paragraphs[:self.max_examples_per_file]:
            if self.min_line_length <= len(para) <= self.max_line_length:
                example = {
                    "text": f"[INST] Explica este contenido:\n{para} [/INST]\n",
                    "source": str(file_path),
                    "type": "markup_content"
                }
                examples.append(example)
        
        return examples
    
    def process_file(self, file_path: Path) -> None:
        """Procesar un archivo y añadir ejemplos al dataset."""
        if not self.should_include_file(file_path):
            logger.debug(f"Skipping {file_path}")
            return
        
        logger.info(f"Processing {file_path}")
        
        examples = self.extract_examples_from_file(file_path)
        self.examples.extend(examples)
        
        logger.info(f"Added {len(examples)} examples from {file_path}")
    
    def process_directory(self, directory_path: Path) -> None:
        """Procesar un directorio recursivamente."""
        all_files = []
        
        for file_path in directory_path.rglob('*'):
            if file_path.is_file():
                all_files.append(file_path)
        
        logger.info(f"Found {len(all_files)} files in {directory_path}")
        
        for file_path in all_files:
            self.process_file(file_path)
    
    def create_dataset(self) -> None:
        """Crear el dataset."""
        if self.input_path.is_file():
            self.process_file(self.input_path)
        elif self.input_path.is_dir():
            self.process_directory(self.input_path)
        else:
            raise FileNotFoundError(f"Input path not found: {self.input_path}")
        
        # Guardar dataset
        self.save_dataset()
    
    def save_dataset(self) -> None:
        """Guardar el dataset en archivo JSONL."""
        output_path = self.output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Saving {len(self.examples)} examples to {output_path}")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for example in self.examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        logger.info(f"Dataset saved successfully to {output_path}")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Create fine-tuning dataset from source code"
    )
    parser.add_argument(
        "input",
        type=str,
        help="Input path (file or directory)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/raw/dataset.jsonl",
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--include",
        type=str,
        nargs="*",
        default=None,
        help="File extensions to include (e.g., .py .js .java)"
    )
    parser.add_argument(
        "--exclude",
        type=str,
        nargs="*",
        default=None,
        help="File extensions to exclude"
    )
    parser.add_argument(
        "--max-file-size",
        type=int,
        default=1024 * 1024,  # 1MB
        help="Maximum file size to process (bytes)"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=50,
        help="Maximum examples per file"
    )
    parser.add_argument(
        "--min-length",
        type=int,
        default=10,
        help="Minimum line length"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=2000,
        help="Maximum line length"
    )
    args = parser.parse_args()
    
    try:
        creator = DatasetCreator(
            input_path=args.input,
            output_path=args.output,
            include_extensions=set(args.include) if args.include else None,
            exclude_extensions=set(args.exclude) if args.exclude else None,
            max_file_size=args.max_file_size,
            max_examples_per_file=args.max_examples,
            min_line_length=args.min_length,
            max_line_length=args.max_length
        )
        
        creator.create_dataset()
        
        logger.info("Dataset creation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
