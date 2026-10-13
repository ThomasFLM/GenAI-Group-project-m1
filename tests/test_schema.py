"""Les fiches de test doivent respecter le schéma XSD officiel 3.5 (docs/schema/3.5)."""
from pathlib import Path

import pytest

etree = pytest.importorskip("lxml.etree")

ROOT = Path(__file__).resolve().parents[1]


def test_fixtures_valid_against_official_schema():
    schema = etree.XMLSchema(etree.parse(str(ROOT / "docs" / "schema" / "3.5" / "Publication.xsd")))
    for f in sorted((ROOT / "tests" / "fixtures").glob("*.xml")):
        assert schema.validate(etree.parse(str(f))), f"{f.name}: {schema.error_log}"
