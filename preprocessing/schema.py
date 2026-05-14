from __future__ import annotations

from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class PathologyAttributes(BaseModel):
    presence: bool = True
    location: Optional[str] = None
    size: Optional[str] = None
    texture: Optional[str] = None
    prominence_score: int = 5  # 0-5
    # 0 = absent, 1 = questionable, 2 = possible, 3 = probable, 4 = likely, 5 = definite


class CanonicalReport(BaseModel):
    uid: int
    findings: List[str] = Field(default_factory=list)
    impression: str = ""
    recommendation: str = ""
    mesh_tags: List[str] = Field(default_factory=list)
    pathology_json: Dict[str, PathologyAttributes] = Field(default_factory=dict)
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
