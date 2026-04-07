from __future__ import annotations
from typing import List, Tuple
from app.knowledge_graph.extraction.schema import Entity, Relation
from app.knowledge_graph.extraction.validator import (
    dedupe_entities,
    dedupe_relations,
    remap_relation_endpoints,
    _build_alias_map,
)


def clean_entities_relations(
    entities: List[Entity], relations: List[Relation]
) -> Tuple[List[Entity], List[Relation]]:
    """
    Full cleaning pipeline:
    1. Merge abbreviation aliases (RNN ↔ Recurrent Neural Network).
    2. Deduplicate entities.
    3. Re-map relation endpoints to canonical names.
    4. Deduplicate relations.
    """
    # Build alias map from raw names BEFORE dedup (preserves both forms)
    alias_map = _build_alias_map([e.name for e in entities if e.name])

    # Deduplicate + canonicalise entities
    entities = dedupe_entities(entities)

    # Remap relation head/tail to canonical names, then dedupe
    relations = remap_relation_endpoints(relations, alias_map)
    relations = dedupe_relations(relations)

    return entities, relations
