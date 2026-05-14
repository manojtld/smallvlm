from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class FindingAttributes(BaseModel):
    severity: Optional[str] = None   # mild | moderate | severe
    location: Optional[str] = None   # bilateral | right lower lobe | etc.
    size: Optional[str] = None       # small | large | cm description


class CanonicalReport(BaseModel):
    uid: int
    findings: List[str] = Field(default_factory=list)
    impression: str = ""
    attributes: Dict[str, FindingAttributes] = Field(default_factory=dict)
    normal: bool = False
    mesh_tags: List[str] = Field(default_factory=list)
    # originals preserved for debugging / fallback
    raw_findings: str = ""
    raw_impression: str = ""


class SFTTask(BaseModel):
    uid: int
    task_type: str  # full_report | findings_only | impression_only | structured_json | normal_classification
    frontal_image: Optional[str] = None
    lateral_image: Optional[str] = None
    prompt: str
    target: str
