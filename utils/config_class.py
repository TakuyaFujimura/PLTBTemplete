from pathlib import Path
from typing import Any, Dict

from pydantic import BaseModel, Field


class CallbackConfig(BaseModel):
    tqdm_refresh_rate: int = 1
    callbacks: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class Config(BaseModel):
    seed: int
    result_dir: Path
    name: str
    version: str

    model: Dict[str, Any]
    trainer: Dict[str, Any]
    callback: CallbackConfig
    datamodule: Dict[str, Any]

