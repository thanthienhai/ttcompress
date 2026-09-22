"""Every Vietnamese-content dataset_source must route to the Vietnamese
prompt template, never silently fall back to PROMPT_TEMPLATES['generic']
(English) -- a real bug caught twice by actually running the pipeline on
real data: a dataset_source with no PROMPT_TEMPLATES entry silently gets the
English generic template instead of erroring. First caught on the
now-removed vcc_bench_v2_needle_in_haystack source; this test exists so
xquad_vi (or any future Vietnamese source) doesn't repeat it silently."""
from ttcompress.reader import DEFAULT_MAX_NEW_TOKENS, PROMPT_TEMPLATES

VIETNAMESE_SOURCES = ('uit_viquad', 'xquad_vi', 'vimqa')


def test_vietnamese_sources_have_explicit_templates():
    for source in VIETNAMESE_SOURCES:
        assert source in PROMPT_TEMPLATES, f"{source} falls back to PROMPT_TEMPLATES['generic'] (English)"


def test_vietnamese_sources_use_the_vietnamese_template():
    for source in VIETNAMESE_SOURCES:
        assert PROMPT_TEMPLATES[source] == PROMPT_TEMPLATES['uit_viquad']
        assert 'tiếng Việt' in PROMPT_TEMPLATES[source]


def test_vietnamese_sources_have_max_new_tokens_entries():
    for source in VIETNAMESE_SOURCES:
        assert source in DEFAULT_MAX_NEW_TOKENS


def test_infinitebench_passkey_has_explicit_template_and_max_new_tokens():
    # Answers are a short digit string -- falling back to 'generic'
    # (English instructions, fine) or the 64-token default (wasteful but not
    # wrong) wouldn't corrupt scores the way a missing Vietnamese template
    # would, but it should still be explicit since the official max_new_tokens
    # (6) is a real, verified value, not a guess.
    assert 'infinitebench_passkey' in PROMPT_TEMPLATES
    assert DEFAULT_MAX_NEW_TOKENS['infinitebench_passkey'] == 6
