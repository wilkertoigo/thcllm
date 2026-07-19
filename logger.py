# ── Logger Module ────────────────────────────────────────────────────────────
"""Configuração de logging independente sem imports circulares"""

import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)

logger = logging.getLogger("THC_LLM")
