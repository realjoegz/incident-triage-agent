"""The fault catalogue as typed objects."""

from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from triage.taxonomy import Category

CATALOGUE_FILE = "catalogue.yaml"

Pair = Annotated[list[str], Field(min_length=2, max_length=2)]


class Service(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    overview: str
    escalation: str


class Fault(BaseModel):
    """A fault the corpus documents: two past incidents and one runbook section."""

    model_config = ConfigDict(extra="forbid")

    id: str
    service: str
    category: Category
    transparency: Literal["transparent", "opaque"]
    title: str
    signature: str
    generic_log: str
    downstream: str
    downstream_log: str
    alert: str
    symptoms: str
    tickets: Pair
    cause: str
    triggers: Pair
    resolution: str
    diagnose: Annotated[list[str], Field(min_length=2)]
    time: str | None = None
    blip: bool = False


class NovelFault(BaseModel):
    """A fault the corpus does not document at all."""

    model_config = ConfigDict(extra="forbid")

    id: str
    service: str
    signature: str
    generic_log: str
    alert: str
    tickets: Annotated[list[str], Field(min_length=1, max_length=1)]


class Catalogue(BaseModel):
    model_config = ConfigDict(extra="forbid")

    services: list[Service]
    faults: list[Fault]
    novel: list[NovelFault]
    noise: list[str]


def load_catalogue(data_dir: Path) -> Catalogue:
    with (data_dir / CATALOGUE_FILE).open(encoding="utf-8") as source:
        return Catalogue.model_validate(yaml.safe_load(source))
