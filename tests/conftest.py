import pytest

PLAN_XML = """<?xml version='1.0' encoding='UTF-8'?>
<timetable station='Köln Hbf'>
  <s id="4124699725257303251-2609231709-15">
    <tl f="N" t="p" o="800337" c="RE" n="10123"/>
    <ar pt="2609231812" pp="5" l="1" ppth="Aachen Hbf|Düren|Köln-Ehrenfeld"/>
    <dp pt="2609231815" pp="5" l="1" ppth="Köln Messe/Deutz|Düsseldorf Hbf"/>
  </s>
  <s id="-771222-2609231800-1">
    <tl f="F" t="p" o="80" c="ICE" n="123"/>
    <dp pt="2609231820" pp="2" ppth="Frankfurt(Main)Hbf"/>
  </s>
</timetable>"""

FCHG_XML = """<?xml version='1.0' encoding='UTF-8'?>
<timetable station="Köln Hbf" eva="8000207">
  <s id="4124699725257303251-2609231709-15" eva="8000207">
    <ar ct="2609231819" l="1"><m id="r1" t="d" c="43" ts="2609231750"/></ar>
    <dp ct="2609231821" cp="6" l="1"/>
  </s>
  <s id="-771222-2609231800-1" eva="8000207">
    <dp cs="c" clt="2609231700"/>
  </s>
  <s id="999-2609231800-3" eva="8000207">
    <m id="h1" t="h" c="0" ts="2609231600"/>
  </s>
</timetable>"""


@pytest.fixture
def plan_xml():
    return PLAN_XML


@pytest.fixture
def fchg_xml():
    return FCHG_XML


def write_parsed(root, responses):
    """Parse (source, eva, collected_at, xml) tuples with the production parser into a
    parsed layer under root/parsed, exactly as the collector would."""
    from nrw_connection_risk.collector.parse import parse_timetable
    from nrw_connection_risk.collector.storage import ParsedStore

    store = ParsedStore(root)
    for source, eva, t, xml in responses:
        store.add(parse_timetable(xml, source, eva, t))
    store.flush()
    return root / "parsed"
