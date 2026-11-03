from iso2json import parse


def test_line_number_ocr_fixes():
    assert parse.fix_line_number("4-ST-7489-A1A") == "4-ST-7489-A1A"
    assert parse.fix_line_number("4-5T-7489-AIA") == "4-ST-7489-A1A"
    assert parse.fix_line_number("garbage") is None


def test_title_block_lines():
    rec = parse.parse_title_block([
        "DWG NO: 1S0-590-20", "LINE NO: 4-ST-7489-A1A", "SIZE: 4 IN", "SPEC: A1A",
        "MATERIAL: CS A106-B", "DESIGN PRESS: 7.0 BARG", "DESIGN TEMP: 65 C", "REV: 0",
    ])
    assert rec["drawing_number"] == "ISO-590-20"
    assert rec["nominal_size_in"] == 4
    assert rec["design_pressure_barg"] == 7.0
    assert rec["design_temp_c"] == 65
    assert rec["revision"] == "0"


def test_title_block_missing_colon():
    rec = parse.parse_title_block(["SPEC A1A"])
    assert rec["pipe_spec"] == "A1A"


def test_tags_and_dims():
    assert parse.parse_tag("HY-315") == "HV-315"  # unique positional prefix match
    assert parse.parse_tag("UT-1541") is None  # ambiguous (LT/FT): do not guess
    assert parse.parse_tag("PSV5988") == "PSV-5988"  # lost dash
    assert parse.parse_tag("hello") is None
    assert parse.parse_dim("4550") == 4550
    assert parse.parse_dim("45") is None


def test_bom_rows():
    rows = parse.parse_bom(["ITEM QTY UNIT DESCRIPTION", "1 2 EA GATE VALVE 4IN |", "2 18 M PIPE SMLS BE 4IN"])
    assert rows[0] == {"item": 1, "qty": 2, "unit": "EA", "description": "GATE VALVE 4IN"}
    assert rows[1]["qty"] == 18
