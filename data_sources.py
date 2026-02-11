"""Dataset registry and language intersection logic for Whisper fine-tuning.

Each source defines its HuggingFace dataset ID, column mappings, split names,
load_dataset kwargs builder, and the full list of supported languages.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

DATA_SOURCES: dict[str, dict] = {
    "mozilla-foundation/common_voice_17_0": {
        "short_name": "common_voice",
        "audio_column": "audio",
        "text_column": "sentence",
        "split_map": {"train": "train", "val": "validation", "test": "test"},
        "load_kwargs": lambda lang: {"name": lang, "trust_remote_code": True},
        "languages": [
            "ab", "af", "am", "an", "ar", "as", "ast", "az", "ba", "bas",
            "be", "bg", "bn", "br", "ca", "ckb", "cnh", "co", "cs", "cv",
            "cy", "da", "de", "dv", "dyu", "el", "en", "eo", "es", "et",
            "eu", "fa", "fi", "fr", "fy-NL", "ga-IE", "gl", "gn", "ha",
            "he", "hi", "hr", "hsb", "hu", "hy-AM", "ia", "id", "ig",
            "is", "it", "ja", "ka", "kab", "kk", "km", "kn", "ko", "ku",
            "ky", "lb", "lg", "lo", "lt", "lv", "mdf", "mg", "mk", "ml",
            "mn", "mr", "mrj", "mt", "my", "myv", "nan-tw", "ne-NP", "nl",
            "nn-NO", "oc", "or", "os", "pa-IN", "pl", "ps", "pt", "quy",
            "rm-sursilv", "rm-vallader", "ro", "ru", "rw", "sah", "sat",
            "sc", "sk", "sl", "sq", "sr", "sv-SE", "sw", "ta", "te", "th",
            "ti", "tig", "tk", "tok", "tr", "tt", "tw", "ug", "uk", "ur",
            "uz", "vi", "vot", "yi", "yo", "yue", "zh-CN", "zh-HK",
            "zh-TW", "zu",
        ],
    },
    "google/fleurs": {
        "short_name": "fleurs",
        "audio_column": "audio",
        "text_column": "transcription",
        "split_map": {"train": "train", "val": "validation", "test": "test"},
        "load_kwargs": lambda lang: {"name": FLEURS_LANG_MAP.get(lang, f"{lang}_{lang}")},
        "languages": [
            "af", "am", "ar", "as", "ast", "az", "be", "bg", "bn", "bs",
            "ca", "ceb", "ckb", "cs", "cy", "da", "de", "el", "en", "es",
            "et", "fa", "fi", "fil", "fr", "ga", "gl", "gu", "ha", "he",
            "hi", "hr", "hu", "hy", "id", "ig", "is", "it", "ja", "jv",
            "ka", "kam", "kea", "kk", "km", "kn", "ko", "ku", "ky", "lb",
            "lg", "ln", "lo", "lt", "luo", "lv", "mi", "mk", "ml", "mn",
            "mr", "ms", "mt", "my", "nb", "ne", "nl", "nn", "ny", "oc",
            "om", "or", "pa", "pl", "ps", "pt", "ro", "ru", "sd", "sk",
            "sl", "sn", "so", "sq", "sr", "sv", "sw", "ta", "te", "tg",
            "th", "tr", "uk", "umb", "ur", "uz", "vi", "wo", "xh", "yo",
            "yue", "zh", "zu",
        ],
    },
    "facebook/voxpopuli": {
        "short_name": "voxpopuli",
        "audio_column": "audio",
        "text_column": "normalized_text",
        "split_map": {"train": "train", "val": "validation", "test": "test"},
        "load_kwargs": lambda lang: {"name": lang, "trust_remote_code": True},
        "languages": [
            "cs", "de", "en", "es", "et", "fi", "fr", "hr", "hu", "it",
            "lt", "nl", "pl", "pt", "ro", "sk", "sl", "sv",
        ],
    },
    "facebook/multilingual_librispeech": {
        "short_name": "mls",
        "audio_column": "audio",
        "text_column": "text",
        "split_map": {"train": "train", "val": "validation", "test": "test"},
        "load_kwargs": lambda lang: {"name": lang, "trust_remote_code": True},
        "languages": ["de", "en", "es", "fr", "it", "nl", "pl", "pt"],
    },
    "issai/Turkish_Speech_Corpus": {
        "short_name": "issai",
        "audio_column": "audio",
        "text_column": "sentence",
        "split_map": {"train": "train", "val": "validation", "test": "test"},
        "load_kwargs": lambda _lang: {},
        "languages": ["tr"],
    },
}

# ---------------------------------------------------------------------------
# FLEURS language code mapping
# FLEURS uses codes like "tr_tr", "de_de", "en_us", etc.
# ---------------------------------------------------------------------------

FLEURS_LANG_MAP: dict[str, str] = {
    "af": "af_za", "am": "am_et", "ar": "ar_eg", "as": "as_in",
    "ast": "ast_es", "az": "az_az", "be": "be_by", "bg": "bg_bg",
    "bn": "bn_in", "bs": "bs_ba", "ca": "ca_es", "ceb": "ceb_ph",
    "ckb": "ckb_iq", "cs": "cs_cz", "cy": "cy_gb", "da": "da_dk",
    "de": "de_de", "el": "el_gr", "en": "en_us", "es": "es_419",
    "et": "et_ee", "fa": "fa_ir", "fi": "fi_fi", "fil": "fil_ph",
    "fr": "fr_fr", "ga": "ga_ie", "gl": "gl_es", "gu": "gu_in",
    "ha": "ha_ng", "he": "he_il", "hi": "hi_in", "hr": "hr_hr",
    "hu": "hu_hu", "hy": "hy_am", "id": "id_id", "ig": "ig_ng",
    "is": "is_is", "it": "it_it", "ja": "ja_jp", "jv": "jv_id",
    "ka": "ka_ge", "kam": "kam_ke", "kea": "kea_cv", "kk": "kk_kz",
    "km": "km_kh", "kn": "kn_in", "ko": "ko_kr", "ku": "ku_arab_iq",
    "ky": "ky_kg", "lb": "lb_lu", "lg": "lg_ug", "ln": "ln_cd",
    "lo": "lo_la", "lt": "lt_lt", "luo": "luo_ke", "lv": "lv_lv",
    "mi": "mi_nz", "mk": "mk_mk", "ml": "ml_in", "mn": "mn_mn",
    "mr": "mr_in", "ms": "ms_my", "mt": "mt_mt", "my": "my_mm",
    "nb": "nb_no", "ne": "ne_np", "nl": "nl_nl", "nn": "nn_no",
    "ny": "ny_mw", "oc": "oc_fr", "om": "om_et", "or": "or_in",
    "pa": "pa_in", "pl": "pl_pl", "ps": "ps_af", "pt": "pt_br",
    "ro": "ro_ro", "ru": "ru_ru", "sd": "sd_in", "sk": "sk_sk",
    "sl": "sl_si", "sn": "sn_zw", "so": "so_so", "sq": "sq_al",
    "sr": "sr_rs", "sv": "sv_se", "sw": "sw_ke", "ta": "ta_in",
    "te": "te_in", "tg": "tg_tj", "th": "th_th", "tr": "tr_tr",
    "uk": "uk_ua", "umb": "umb_ao", "ur": "ur_pk", "uz": "uz_uz",
    "vi": "vi_vn", "wo": "wo_sn", "xh": "xh_za", "yo": "yo_ng",
    "yue": "yue_hant_hk", "zh": "cmn_hans_cn", "zu": "zu_za",
}

# ---------------------------------------------------------------------------
# Common Voice language code normalization
# CV uses codes like "fy-NL", "ga-IE", etc. Map simple codes to CV codes.
# ---------------------------------------------------------------------------

CV_LANG_MAP: dict[str, str] = {
    "fy": "fy-NL", "ga": "ga-IE", "hy": "hy-AM", "ne": "ne-NP",
    "nn": "nn-NO", "pa": "pa-IN", "sv": "sv-SE",
}


def _normalize_cv_lang(lang: str) -> str:
    """Return Common Voice language code for a given ISO code."""
    return CV_LANG_MAP.get(lang, lang)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_sources_for_language(lang: str) -> list[dict]:
    """Return dataset source configs available for the given language.

    Args:
        lang: ISO 639-1 language code (e.g. "tr", "de", "en").

    Returns:
        List of dicts with keys: dataset_id, short_name, audio_column,
        text_column, split_map, load_kwargs.
    """
    results = []
    for dataset_id, src in DATA_SOURCES.items():
        # Check both the raw code and normalized variants
        langs = src["languages"]
        matches = (
            lang in langs
            or _normalize_cv_lang(lang) in langs
        )
        if matches:
            load_lang = lang
            if dataset_id == "mozilla-foundation/common_voice_17_0":
                load_lang = _normalize_cv_lang(lang)
            results.append({
                "dataset_id": dataset_id,
                "short_name": src["short_name"],
                "audio_column": src["audio_column"],
                "text_column": src["text_column"],
                "split_map": src["split_map"],
                "load_kwargs": src["load_kwargs"](load_lang),
            })
    return results


def list_all_languages() -> dict[str, list[str]]:
    """List all languages and which sources support each.

    Returns:
        Dict mapping language code to list of short source names.
        Example: {"tr": ["common_voice", "fleurs", "issai"], ...}
    """
    lang_map: dict[str, list[str]] = {}
    for src in DATA_SOURCES.values():
        for lang in src["languages"]:
            # Normalize CV-style codes back to simple form for display
            simple = lang.split("-")[0] if "-" in lang else lang
            lang_map.setdefault(simple, [])
            if src["short_name"] not in lang_map[simple]:
                lang_map[simple].append(src["short_name"])
    return dict(sorted(lang_map.items()))


if __name__ == "__main__":
    langs = list_all_languages()
    print(f"{'Language':<10} {'Sources':>3}  Details")
    print("-" * 60)
    for lang, sources in langs.items():
        print(f"{lang:<10} {len(sources):>3}  {', '.join(sources)}")
    print(f"\nTotal: {len(langs)} languages")
