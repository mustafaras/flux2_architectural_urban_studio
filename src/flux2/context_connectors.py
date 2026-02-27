from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(slots=True)
class ConnectorReference:
    reference_name: str
    category: str
    tags: list[str]
    source: str
    notes: str


class ExternalContextConnector(Protocol):
    connector_id: str

    def fetch_context(self, query: str, project_context: dict[str, Any]) -> list[ConnectorReference]:
        ...


class MockGISConnector:
    connector_id = "mock-gis"

    def fetch_context(self, query: str, project_context: dict[str, Any]) -> list[ConnectorReference]:
        geography = str(project_context.get("geography", "unknown"))
        label = query.strip() or geography
        return [
            ConnectorReference(
                reference_name=f"{label} mobility corridor stub",
                category="aerial/context image",
                tags=["connector", "gis", "mobility", geography],
                source="mock-gis://context",
                notes="Stub connector payload for phase-gated integration.",
            ),
            ConnectorReference(
                reference_name=f"{label} public realm stub",
                category="precedent board",
                tags=["connector", "gis", "public-realm", geography],
                source="mock-gis://public-realm",
                notes="Stub typology context result.",
            ),
        ]


def available_connectors() -> dict[str, ExternalContextConnector]:
    return {
        MockGISConnector.connector_id: MockGISConnector(),
    }


def run_connector(
    *,
    connector_id: str,
    query: str,
    project_context: dict[str, Any],
) -> list[dict[str, Any]]:
    connector = available_connectors().get(connector_id)
    if connector is None:
        return []

    results = connector.fetch_context(query=query, project_context=project_context)
    out: list[dict[str, Any]] = []
    for item in results:
        out.append(
            {
                "name": item.reference_name,
                "category": item.category,
                "tags": item.tags,
                "source": item.source,
                "notes": item.notes,
            }
        )
    return out
