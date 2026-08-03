"""Hidden tests — bpe_merge_order.

Every assertion is licensed by a sentence in prompt.md. No tolerances here:
tokenization is exact or it is broken.
"""

import pytest

from bpe_encode import decode, encode


# -- the byte-level starting point ----------------------------------------


def test_no_merges_leaves_one_token_per_byte():
    assert encode("abc", []) == [b"a", b"b", b"c"]


def test_the_starting_point_is_bytes_not_characters():
    # "é" is two bytes in UTF-8, so it starts life as two tokens.
    assert encode("é", []) == [b"\xc3", b"\xa9"]


def test_a_four_byte_character_starts_as_four_tokens():
    assert encode("🙂", []) == [b"\xf0", b"\x9f", b"\x99", b"\x82"]


def test_empty_text_encodes_to_nothing():
    assert encode("", [(b"a", b"b")]) == []


def test_tokens_come_back_as_bytes_objects():
    assert all(isinstance(token, bytes) for token in encode("hello", [(b"l", b"l")]))


# -- the ordering, which is the whole task --------------------------------


def test_rank_beats_position():
    # Left to right, (a, b) comes first. By rank, (b, c) comes first.
    merges = [(b"b", b"c"), (b"a", b"b")]
    assert encode("abc", merges) == [b"a", b"bc"]


def test_rank_beats_frequency():
    # (a, b) occurs twice and (b, a) once, but (b, a) was learned first.
    merges = [(b"b", b"a"), (b"a", b"b")]
    assert encode("abab", merges) == [b"a", b"ba", b"b"]


def test_a_later_merge_applies_to_what_an_earlier_one_built():
    merges = [(b"a", b"b"), (b"ab", b"c")]
    assert encode("abc", merges) == [b"abc"]


def test_a_merge_only_fires_once_its_inputs_exist():
    # (ab, c) is rank 0 but nothing spells "ab" until (a, b) has run.
    merges = [(b"ab", b"c"), (b"a", b"b")]
    assert encode("abc", merges) == [b"abc"]


def test_several_rounds_in_order():
    merges = [(b"a", b"n"), (b"an", b"an"), (b"b", b"anan")]
    assert encode("banana", merges) == [b"banan", b"a"]


# -- applying one merge across the whole sequence -------------------------


def test_every_occurrence_of_the_winning_pair_is_merged():
    assert encode("abab", [(b"a", b"b")]) == [b"ab", b"ab"]


def test_repeated_bytes_merge_left_to_right_without_overlapping():
    assert encode("aaa", [(b"a", b"a")]) == [b"aa", b"a"]


def test_the_leftover_of_an_overlap_can_merge_in_a_later_round():
    assert encode("aaaa", [(b"a", b"a"), (b"aa", b"aa")]) == [b"aaaa"]


def test_five_of_a_kind():
    assert encode("aaaaa", [(b"a", b"a")]) == [b"aa", b"aa", b"a"]


# -- boundaries the tokenizer does not have -------------------------------


def test_merges_cross_whitespace():
    assert encode("a the", [(b" ", b"t")]) == [b"a", b" t", b"h", b"e"]


def test_merges_cross_punctuation():
    assert encode("a,b", [(b",", b"b")]) == [b"a", b",b"]


def test_merges_can_join_the_bytes_inside_one_character():
    tokens = encode("é", [(b"\xc3", b"\xa9")])
    assert tokens == [b"\xc3\xa9"]
    assert decode(tokens) == "é"


# -- merges that do not apply ---------------------------------------------


def test_unused_merges_are_harmless():
    assert encode("abc", [(b"z", b"q"), (b"x", b"y")]) == [b"a", b"b", b"c"]


def test_a_single_byte_of_text_has_no_pairs():
    assert encode("a", [(b"a", b"a")]) == [b"a"]


def test_the_merge_list_is_left_alone():
    merges = [(b"a", b"b"), (b"ab", b"c")]
    before = list(merges)
    encode("abcabc", merges)
    assert merges == before


# -- decoding --------------------------------------------------------------


def test_decode_joins_tokens_back_into_text():
    assert decode([b"ban", b"an", b"a"]) == "banana"


def test_decode_of_nothing_is_the_empty_string():
    assert decode([]) == ""


def test_decode_reassembles_a_character_split_across_tokens():
    assert decode([b"\xc3", b"\xa9"]) == "é"


@pytest.mark.parametrize(
    "text",
    [
        "banana",
        "the quick brown fox",
        "  leading and trailing  ",
        "héllo wörld",
        "🙂 mixed 🙃 emoji",
        "newlines\nand\ttabs",
    ],
)
def test_round_trip_is_exact(text):
    merges = [
        (b"t", b"h"),
        (b"th", b"e"),
        (b"a", b"n"),
        (b"an", b"a"),
        (b" ", b"w"),
        (b"o", b"r"),
        (b"\xc3", b"\xa9"),
        (b"\xf0", b"\x9f"),
    ]
    assert decode(encode(text, merges)) == text


def test_a_longer_passage_survives_a_bigger_merge_table():
    text = "the fox and the hound, the end. " * 20
    merges = [
        (b"t", b"h"),
        (b"th", b"e"),
        (b"the", b" "),
        (b"a", b"n"),
        (b"an", b"d"),
        (b" ", b"f"),
        (b"o", b"x"),
        (b"h", b"o"),
        (b"ho", b"u"),
        (b"hou", b"n"),
    ]
    tokens = encode(text, merges)
    assert decode(tokens) == text
    # The table is doing real work, not passing bytes through.
    assert len(tokens) < len(text.encode("utf-8"))
