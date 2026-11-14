from search import BM25, rrf, tokenize


def test_tokenize_lowercases_drops_stopwords_and_keeps_number_units():
    assert tokenize("Take 0.6 mg of Colchicine, then the next dose") == \
        ["take", "0.6", "mg", "0.6mg", "colchicine", "next", "dose"]
    assert tokenize("0.6mg") == ["0.6", "mg", "0.6mg"]
    assert tokenize("?? the and of") == []


def test_bm25_ranks_exact_term_first():
    bm = BM25({1: "drink water to prevent kidney stones",
               2: "colchicine 0.6 mg missed dose take when you remember",
               3: "gout is a kind of arthritis"})
    s = bm.scores("missed colchicine dose")
    assert max(s, key=s.get) == 2 and 1 not in s


def test_bm25_ids_limit_scoring_but_not_statistics():
    docs = {1: "colchicine dose", 2: "colchicine label", 3: "water"}
    assert set(BM25(docs).scores("colchicine", ids={2})) == {2}
    assert BM25(docs).scores("colchicine", ids={2})[2] == BM25(docs).scores("colchicine")[2]


def test_bm25_empty_query_and_empty_corpus():
    assert BM25({1: "water"}).scores("?? the") == {}
    assert BM25({}).scores("water") == {}


def test_bm25_allows_zero_length_document():
    s = BM25({1: "", 2: "the of", 3: "water"}).scores("water")
    assert list(s) == [3] and s[3] > 0


def test_rrf_merges_and_dedupes():
    assert rrf([1, 2, 3], [3, 4]) == [3, 1, 2, 4]
    assert rrf([], [5]) == [5]
