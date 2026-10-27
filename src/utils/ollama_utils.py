"""
Utilidades de Ollama para LLM-FineTuner.

Proporciona funciones para interactuar con Ollama CLI.
"""

import json
import logging
import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


@dataclass
class OllamaModelInfo:
    """Información sobre un modelo en Ollama."""
    name: str
    model_id: str
    size: str
    digest: str
    modified_at: str
    size_bytes: int = 0
    
    @classmethod
    def from_list_output(cls, line: str) -> Optional["OllamaModelInfo"]:
        """Crear desde línea de salida de `ollama list`."""
        parts = line.strip().split()
        if len(parts) >= 4:
            try:
                size_str = parts[1]
                # Convertir tamaño a bytes
                size_bytes = 0
                if size_str.endswith("GB"):
                    size_bytes = int(float(size_str[:-2]) * 1024 ** 3)
                elif size_str.endswith("MB"):
                    size_bytes = int(float(size_str[:-2]) * 1024 ** 2)
                elif size_str.endswith("KB"):
                    size_bytes = int(float(size_str[:-2]) * 1024)
                elif size_str.endswith("B"):
                    size_bytes = int(size_str[:-1])
                
                return cls(
                    name=parts[0],
                    model_id=parts[0],
                    size=size_str,
                    digest=parts[2] if len(parts) >= 3 else "",
                    modified_at=" ".join(parts[3:]) if len(parts) >= 4 else "",
                    size_bytes=size_bytes
                )
            except (ValueError, IndexError):
                pass
        return None
    
    def to_dict(self) -> Dict:
        """Convertir a diccionario."""
        return {
            "name": self.name,
            "model_id": self.model_id,
            "size": self.size,
            "size_bytes": self.size_bytes,
            "digest": self.digest,
            "modified_at": self.modified_at
        }


@dataclass
class OllamaVersion:
    """Información de versión de Ollama."""
    version: str
    commit: Optional[str] = None
    
    @classmethod
    def from_version_output(cls, output: str) -> "OllamaVersion":
        """Crear desde salida de `ollama --version`."""
        lines = output.strip().split('\n')
        version = lines[0].strip() if lines else "unknown"
        commit = lines[1].strip() if len(lines) > 1 else None
        return cls(version=version, commit=commit)


def find_ollama_executable() -> Optional[str]:
    """
    Encontrar el ejecutable de Ollama.
    
    Busca en:
    1. PATH del sistema
    2. Ruta por defecto de Windows
    3. Ruta por defecto de macOS/Linux
    
    Returns:
        Ruta al ejecutable o None si no se encuentra
    """
    # Verificar en PATH
    import shutil
    ollama = shutil.which("ollama")
    if ollama:
        return ollama
    
    # Windows paths
    windows_paths = [
        "C:/Program Files/Ollama/ollama.exe",
        "C:/Program Files (x86)/Ollama/ollama.exe",
        "C:/Ollama/ollama.exe",
    ]
    
    # macOS paths
    macos_paths = [
        "/usr/local/bin/ollama",
        "/opt/homebrew/bin/ollama",
        "~/Library/Application Support/ollama/ollama",
    ]
    
    # Linux paths
    linux_paths = [
        "/usr/bin/ollama",
        "/usr/local/bin/ollama",
        "~/bin/ollama",
    ]
    
    # Buscar según sistema operativo
    if os.name == "nt":  # Windows
        for path in windows_paths:
            if os.path.exists(path):
                return path
    elif os.name == "posix":
        if sys.platform == "darwin":  # macOS
            for path in macos_paths:
                expanded = os.path.expanduser(path)
                if os.path.exists(expanded):
                    return expanded
        else:  # Linux
            for path in linux_paths:
                expanded = os.path.expanduser(path)
                if os.path.exists(expanded):
                    return expanded
    
    return None


def run_ollama_command(
    command: List[str],
    timeout: int = 30,
    cwd: Optional[str] = None
) -> Tuple[int, str, str]:
    """
    Ejecutar comando de Ollama.
    
    Args:
        command: Lista de argumentos para el comando
        timeout: Timeout en segundos
        cwd: Directorio de trabajo
        
    Returns:
        Tuple (returncode, stdout, stderr)
    """
    import subprocess
    
    # Buscar ejecutable
    ollama_path = find_ollama_executable()
    if not ollama_path:
        raise FileNotFoundError("Ollama executable not found. Please install Ollama.")
    
    full_command = [ollama_path] + command
    
    try:
        result = subprocess.run(
            full_command,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd
        )
        return result.returncode, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        logger.error(f"Command timed out: {' '.join(full_command)}")
        return -1, "", "Command timed out"
    except Exception as e:
        logger.error(f"Error running command: {e}")
        return -1, "", str(e)


def check_ollama_installed() -> bool:
    """
    Verificar si Ollama está instalado.
    
    Returns:
        True si Ollama está instalado y accesible
    """
    try:
        ollama_path = find_ollama_executable()
        if not ollama_path:
            return False
        
        returncode, stdout, stderr = run_ollama_command(["--version"])
        return returncode == 0
    except Exception:
        return False


def get_ollama_version() -> Optional[OllamaVersion]:
    """
    Obtener versión de Ollama.
    
    Returns:
        OllamaVersion o None si falla
    """
    try:
        returncode, stdout, stderr = run_ollama_command(["--version"], timeout=10)
        if returncode == 0:
            return OllamaVersion.from_version_output(stdout)
    except Exception as e:
        logger.error(f"Error getting Ollama version: {e}")
    return None


def get_ollama_models(local_only: bool = False) -> List[OllamaModelInfo]:
    """
    Obtener lista de modelos disponibles en Ollama.
    
    Args:
        local_only: Solo modelos locales
        
    Returns:
        Lista de OllamaModelInfo
    """
    models = []
    
    try:
        if local_only:
            returncode, stdout, stderr = run_ollama_command(["list"], timeout=30)
        else:
            returncode, stdout, stderr = run_ollama_command(["list", "--all"], timeout=30)
        
        if returncode != 0:
            logger.error(f"Error listing Ollama models: {stderr}")
            return models
        
        for line in stdout.strip().split('\n')[1:]:  # Saltar encabezado
            if line.strip():
                model_info = OllamaModelInfo.from_list_output(line)
                if model_info:
                    models.append(model_info)
    except Exception as e:
        logger.error(f"Error getting Ollama models: {e}")
    
    return models


def verify_model_in_ollama(model_name: str) -> bool:
    """
    Verificar si un modelo existe en Ollama.
    
    Args:
        model_name: Nombre del modelo
        
    Returns:
        True si el modelo existe
    """
    models = get_ollama_models(local_only=True)
    return any(m.name == model_name or m.model_id == model_name for m in models)


def pull_model(model_name: str, timeout: int = 300) -> bool:
    """
    Descargar un modelo con Ollama.
    
    Args:
        model_name: Nombre del modelo a descargar
        timeout: Timeout en segundos
        
    Returns:
        True si se descargó correctamente
    """
    try:
        logger.info(f"Pulling model '{model_name}'...")
        returncode, stdout, stderr = run_ollama_command(["pull", model_name], timeout=timeout)
        
        if returncode == 0:
            logger.info(f"Model '{model_name}' pulled successfully")
            return True
        else:
            logger.error(f"Error pulling model: {stderr}")
            return False
    except Exception as e:
        logger.error(f"Error pulling model: {e}")
        return False


def create_model_from_modelfile(
    model_name: str,
    modelfile_path: str,
    timeout: int = 300
) -> bool:
    """
    Crear modelo en Ollama desde Modelfile.
    
    Args:
        model_name: Nombre del modelo
        modelfile_path: Ruta al Modelfile
        timeout: Timeout en segundos
        
    Returns:
        True si se creó correctamente
    """
    try:
        logger.info(f"Creating Ollama model '{model_name}' from {modelfile_path}...")
        
        modelfile_path = os.path.abspath(modelfile_path)
        modelfile_dir = os.path.dirname(modelfile_path)
        
        returncode, stdout, stderr = run_ollama_command(
            ["create", model_name, "-f", modelfile_path],
            timeout=timeout,
            cwd=modelfile_dir
        )
        
        if returncode == 0:
            logger.info(f"Model '{model_name}' created successfully")
            return True
        else:
            logger.error(f"Error creating model: {stderr}")
            return False
    except Exception as e:
        logger.error(f"Error creating model: {e}")
        return False


def delete_model(model_name: str, timeout: int = 30) -> bool:
    """
    Eliminar modelo de Ollama.
    
    Args:
        model_name: Nombre del modelo
        timeout: Timeout en segundos
        
    Returns:
        True si se eliminó correctamente
    """
    try:
        logger.info(f"Deleting model '{model_name}'...")
        returncode, stdout, stderr = run_ollama_command(["rm", model_name], timeout=timeout)
        
        if returncode == 0:
            logger.info(f"Model '{model_name}' deleted successfully")
            return True
        else:
            logger.error(f"Error deleting model: {stderr}")
            return False
    except Exception as e:
        logger.error(f"Error deleting model: {e}")
        return False


def get_model_info_from_ollama(model_name: str) -> Optional[Dict]:
    """
    Obtener información detallada de un modelo en Ollama.
    
    Args:
        model_name: Nombre del modelo
        
    Returns:
        Diccionario con información o None
    """
    try:
        returncode, stdout, stderr = run_ollama_command(
            ["show", model_name],
            timeout=30
        )
        
        if returncode == 0:
            # Parsing básico de la salida
            info = {"name": model_name}
            for line in stdout.split('\n'):
                line = line.strip()
                if line.startswith("Model size:"):
                    info["size"] = line.split(":", 1)[1].strip()
                elif line.startswith("Model file:"):
                    info["file"] = line.split(":", 1)[1].strip()
                elif line.startswith("Modified:"):
                    info["modified"] = line.split(":", 1)[1].strip()
                elif line.startswith("Digest:"):
                    info["digest"] = line.split(":", 1)[1].strip()
            
            return info
        else:
            logger.error(f"Error getting model info: {stderr}")
            return None
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        return None


def check_ollama_running() -> bool:
    """
    Verificar si el servidor de Ollama está en ejecución.
    
    Returns:
        True si el servidor está en ejecución
    """
    try:
        # Verificar si el puerto está abierto
        import socket
        
        # Puerto por defecto de Ollama
        port = 11434
        timeout = 1
        
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=timeout):
                return True
        except (OSError, socket.timeout):
            return False
    except Exception as e:
        logger.error(f"Error checking Ollama server: {e}")
        return False


def start_ollama_server() -> bool:
    """
    Iniciar el servidor de Ollama.
    
    Returns:
        True si se inició correctamente
    """
    try:
        logger.info("Starting Ollama server...")
        returncode, stdout, stderr = run_ollama_command(["serve"], timeout=10)
        
        # El servidor se ejecuta en segundo plano
        # Verificar si ya está en ejecución
        return check_ollama_running()
    except Exception as e:
        logger.error(f"Error starting Ollama server: {e}")
        return False


def stop_ollama_server() -> bool:
    """
    Detener el servidor de Ollama.
    
    Returns:
        True si se detuvo correctamente
    """
    # Ollama no tiene comando para detener el servidor directamente
    # Hay que usar el sistema operativo
    try:
        import psutil
        
        ollama_path = find_ollama_executable()
        if not ollama_path:
            logger.error("Ollama executable not found")
            return False
        
        # Buscar procesos de Ollama
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if proc.info['name'] and 'ollama' in proc.info['name'].lower():
                    logger.info(f"Killing Ollama process {proc.info['pid']}")
                    proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        
        return True
    except ImportError:
        logger.error("psutil not installed. Cannot stop Ollama server.")
        return False
    except Exception as e:
        logger.error(f"Error stopping Ollama server: {e}")
        return False


def generate_modelfile_content(
    model_name: str,
    model_file: str,
    from_model: str,
    parameters: Optional[Dict[str, str]] = None
) -> str:
    """
    Generar contenido de Modelfile para Ollama.
    
    Args:
        model_name: Nombre del modelo
        model_file: Nombre del archivo GGUF
        from_model: Modelo base (ej: "qwen2.5", "mistral", "llama")
        parameters: Parámetros adicionales
        
    Returns:
        Contenido del Modelfile
    """
    # Parámetros por defecto para diferentes modelos
    default_parameters = {
        "qwen2.5": {
            "num_ctx": "4096",
            "num_gqa": "8",
            "num_gpu": "99",
            "num_thread": "8",
        },
        "qwen2": {
            "num_ctx": "4096",
            "num_head": "32",
            "num_gpu": "99",
            "num_thread": "8",
        },
        "mistral": {
            "num_ctx": "4096",
            "num_gqa": "8",
            "num_gpu": "99",
            "num_thread": "8",
        },
        "llama": {
            "num_ctx": "4096",
            "num_head": "32",
            "num_kv_head": "8",
            "num_gpu": "99",
            "num_thread": "8",
        },
        "llama3": {
            "num_ctx": "4096",
            "num_head": "32",
            "num_kv_head": "8",
            "num_gpu": "99",
            "num_thread": "8",
        },
        "gemma": {
            "num_ctx": "4096",
            "num_gqa": "8",
            "num_gpu": "99",
            "num_thread": "8",
        }
    }
    
    # Obtener parámetros base
    base_params = default_parameters.get(from_model, default_parameters["qwen2.5"])
    
    # Fusionar con parámetros personalizados
    all_params = {**base_params, **(parameters or {})}
    
    # Generar Modelfile
    content = f"""FROM {from_model}

"""
    
    # Añadir metadatos
    content += """# Model metadata
LABEL author "LLM-FineTuner"
LABEL description "Fine-tuned model generated by LLM-FineTuner"

"""
    
    # Añadir archivo del modelo
    content += f"PARAMETER model {model_file}\n\n"
    
    # Añadir parámetros
    content += "# Configuration parameters\n"
    for key, value in all_params.items():
        content += f"PARAMETER {key} {value}\n"
    
    content += """
# License
LICENSE MIT
"""
    
    return content
