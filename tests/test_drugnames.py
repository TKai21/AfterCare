import json

from config import PROJECT_DIR
from drugnames import DrugLexicon, get_lexicon

DATA = PROJECT_DIR / "data"


def test_every_food_table_nutrient_is_excluded():
    lex = get_lexicon()
    foods = json.loads((DATA / "foods.json").read_text())["foods"]
    keys = {k for f in foods for k in f["nutrients"]}
    assert "sodium_mg" in keys
    for key in keys | {"vitamin_d_mcg", "vitamin_k_mcg", "vitamin_c_mg"}:
        name = key.rsplit("_", 1)[0].replace("_", " ")
        assert lex.find(name) == set(), name


def test_measured_collisions_are_excluded():
    text = "drink water, limit sodium, take vitamin D, iron and zinc; Today I feel ok; alli"
    assert get_lexicon().find(text) == set()


def test_real_spelling_is_found():
    lex = get_lexicon()
    assert lex.find("tylenol") == {"tylenol"}
    assert lex.find("Advil's") == {"advil"}
    assert lex.find("ADVIL") == {"advil"}
    assert lex.find("two aspirins") == {"aspirin"}


def test_brand_and_ingredient_equivalents():
    lex = get_lexicon()
    assert "acetaminophen" in lex.equivalents("tylenol")
    assert "colcrys" in lex.equivalents("colchicine")
    assert "tylenol" in lex.equivalents("Tylenol")


def test_whole_words_only():
    lex = get_lexicon()
    assert lex.find("naproxenate") == set()
    assert lex.find("colchicines are") == {"colchicine"}


def test_names_for_rxcuis():
    lex = get_lexicon()
    rxcui = lex.names["aspirin"]["rxcui"]
    assert "aspirin" in lex.names_for_rxcuis([rxcui])
    assert lex.names_for_rxcuis(["0"]) == set()


def test_exclusions_drop_names(tmp_path):
    data = tmp_path / "d.json"
    data.write_text(json.dumps({"names": {
        "water": {"rxcui": "1", "tty": "IN", "ingredients": ["1"]},
        "advil": {"rxcui": "2", "tty": "BN", "ingredients": ["3"]},
        "ibuprofen": {"rxcui": "3", "tty": "IN", "ingredients": ["3"]}}}))
    ex = tmp_path / "e.json"
    ex.write_text(json.dumps({"nutrients": [], "words": ["water"]}))
    lex = DrugLexicon.load(data, ex)
    assert lex.find("water and Advil") == {"advil"}
    assert lex.equivalents("advil") == {"advil", "ibuprofen"}


def test_aftercare_text_has_no_drug_hits():
    lex = get_lexicon()
    for text in [
        "drink plenty of water", "limit sodium to 2000 mg",
        "oxalate like spinach, beets, nuts, rhubarb", "coffee and tea",
        "lemon juice, orange juice, cranberry", "cherries", "honey", "fish oil",
        "fever, rest, ice, heat", "your creatinine and urea levels",
        "Julie from the clinic will call", "turkey has tryptophan",
        "apply petrolatum or Vaseline to the wound",
    ]:
        assert lex.find(text) == set(), text
