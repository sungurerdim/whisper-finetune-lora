#!/usr/bin/env python3
"""Interactive CLI for Whisper LoRA fine-tuning pipeline.

Provides a rich terminal menu to run each pipeline step or the full pipeline,
with language search, config display, and result viewing.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text

console = Console()

# ---------------------------------------------------------------------------
# Language names for display (ISO 639-1 subset used by Whisper datasets)
# ---------------------------------------------------------------------------

LANGUAGE_NAMES: dict[str, str] = {
    "ab": "Abkhaz", "af": "Afrikaans", "am": "Amharic", "an": "Aragonese",
    "ar": "Arabic", "as": "Assamese", "ast": "Asturian", "az": "Azerbaijani",
    "ba": "Bashkir", "bas": "Basaa", "be": "Belarusian", "bg": "Bulgarian",
    "bn": "Bengali", "br": "Breton", "bs": "Bosnian", "ca": "Catalan",
    "ceb": "Cebuano", "ckb": "Central Kurdish", "cnh": "Chin Haka",
    "co": "Corsican", "cs": "Czech", "cv": "Chuvash", "cy": "Welsh",
    "da": "Danish", "de": "German", "dv": "Dhivehi", "dyu": "Dyula",
    "el": "Greek", "en": "English", "eo": "Esperanto", "es": "Spanish",
    "et": "Estonian", "eu": "Basque", "fa": "Persian", "fi": "Finnish",
    "fil": "Filipino", "fr": "French", "fy": "Frisian", "ga": "Irish",
    "gl": "Galician", "gn": "Guarani", "gu": "Gujarati", "ha": "Hausa",
    "he": "Hebrew", "hi": "Hindi", "hr": "Croatian", "hsb": "Upper Sorbian",
    "hu": "Hungarian", "hy": "Armenian", "ia": "Interlingua", "id": "Indonesian",
    "ig": "Igbo", "is": "Icelandic", "it": "Italian", "ja": "Japanese",
    "jv": "Javanese", "ka": "Georgian", "kab": "Kabyle", "kam": "Kamba",
    "kea": "Kabuverdianu", "kk": "Kazakh", "km": "Khmer", "kn": "Kannada",
    "ko": "Korean", "ku": "Kurdish", "ky": "Kyrgyz", "lb": "Luxembourgish",
    "lg": "Luganda", "ln": "Lingala", "lo": "Lao", "lt": "Lithuanian",
    "luo": "Luo", "lv": "Latvian", "mdf": "Moksha", "mg": "Malagasy",
    "mi": "Maori", "mk": "Macedonian", "ml": "Malayalam", "mn": "Mongolian",
    "mr": "Marathi", "mrj": "Western Mari", "ms": "Malay", "mt": "Maltese",
    "my": "Burmese", "myv": "Erzya", "nb": "Norwegian Bokmal",
    "ne": "Nepali", "nl": "Dutch", "nn": "Norwegian Nynorsk", "ny": "Chichewa",
    "oc": "Occitan", "om": "Oromo", "or": "Odia", "os": "Ossetian",
    "pa": "Punjabi", "pl": "Polish", "ps": "Pashto", "pt": "Portuguese",
    "quy": "Quechua", "rm": "Romansh", "ro": "Romanian", "ru": "Russian",
    "rw": "Kinyarwanda", "sah": "Sakha", "sat": "Santali", "sc": "Sardinian",
    "sd": "Sindhi", "sk": "Slovak", "sl": "Slovenian", "sn": "Shona",
    "so": "Somali", "sq": "Albanian", "sr": "Serbian", "sv": "Swedish",
    "sw": "Swahili", "ta": "Tamil", "te": "Telugu", "tg": "Tajik",
    "th": "Thai", "ti": "Tigrinya", "tig": "Tigre", "tk": "Turkmen",
    "tok": "Toki Pona", "tr": "Turkish", "tt": "Tatar", "tw": "Twi",
    "ug": "Uyghur", "uk": "Ukrainian", "umb": "Umbundu", "ur": "Urdu",
    "uz": "Uzbek", "vi": "Vietnamese", "vot": "Votic", "wo": "Wolof",
    "xh": "Xhosa", "yi": "Yiddish", "yo": "Yoruba", "yue": "Cantonese",
    "zh": "Chinese", "zu": "Zulu",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration file."""
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def display_config(config: dict) -> None:
    """Display configuration as a rich table."""
    table = Table(title="config.yaml", show_header=True, header_style="bold cyan")
    table.add_column("Bolum", style="bold")
    table.add_column("Parametre")
    table.add_column("Deger", style="green")

    for section, values in config.items():
        if isinstance(values, dict):
            for key, val in values.items():
                table.add_row(section, key, str(val))
        else:
            table.add_row("", section, str(values))

    console.print(table)


def get_language_map() -> dict[str, list[str]]:
    """Get available languages and their sources from data_sources module."""
    try:
        from data_sources import list_all_languages
        return list_all_languages()
    except ImportError:
        console.print("[red]data_sources.py bulunamadi![/red]")
        return {}


def select_language() -> str | None:
    """Interactive language selection with search.

    Returns:
        Selected ISO 639-1 language code, or None if cancelled.
    """
    lang_map = get_language_map()
    if not lang_map:
        return None

    while True:
        query = Prompt.ask(
            "\n[bold]Dil kodu veya isim girin[/bold] (orn: tr, turkish, german; "
            "'list' = tumu, 'q' = iptal)"
        ).strip().lower()

        if query == "q":
            return None

        if query == "list":
            _print_language_table(lang_map, lang_map)
            continue

        # Filter by code or name
        matches: dict[str, list[str]] = {}
        for code, sources in lang_map.items():
            name = LANGUAGE_NAMES.get(code, "")
            if query == code or query in name.lower() or query in code:
                matches[code] = sources

        if not matches:
            console.print(f"[yellow]'{query}' icin eslesme bulunamadi. Tekrar deneyin.[/yellow]")
            continue

        if len(matches) == 1:
            code = next(iter(matches))
            name = LANGUAGE_NAMES.get(code, code)
            sources = matches[code]
            console.print(
                f"[green]Secildi:[/green] {name} ({code}) "
                f"- Kaynaklar: {', '.join(sources)}"
            )
            return code

        # Multiple matches - show table and let user pick
        _print_language_table(matches, lang_map)
        pick = Prompt.ask("[bold]Dil kodu secin[/bold]").strip().lower()
        if pick in matches:
            name = LANGUAGE_NAMES.get(pick, pick)
            console.print(
                f"[green]Secildi:[/green] {name} ({pick}) "
                f"- Kaynaklar: {', '.join(matches[pick])}"
            )
            return pick
        console.print(f"[yellow]'{pick}' listede yok. Tekrar deneyin.[/yellow]")


def _print_language_table(
    matches: dict[str, list[str]],
    full_map: dict[str, list[str]],
) -> None:
    """Print a table of languages with their sources."""
    table = Table(show_header=True, header_style="bold cyan")
    table.add_column("Kod", style="bold", width=6)
    table.add_column("Dil")
    table.add_column("Kaynaklar")
    table.add_column("Kaynak Sayisi", justify="right")

    for code, sources in matches.items():
        name = LANGUAGE_NAMES.get(code, code)
        table.add_row(code, name, ", ".join(sources), str(len(sources)))

    console.print(table)
    console.print(f"[dim]{len(matches)} dil listelendi[/dim]")


def run_step(label: str, cmd_args: list[str]) -> bool:
    """Run a pipeline step via subprocess.

    Args:
        label: Display label for the step.
        cmd_args: Command arguments (without python executable).

    Returns:
        True if step succeeded.
    """
    full_cmd = [sys.executable] + cmd_args
    console.print(f"\n[bold blue]>>> {label}[/bold blue]")
    console.print(f"[dim]$ {' '.join(full_cmd)}[/dim]\n")

    result = subprocess.run(full_cmd)

    if result.returncode == 0:
        console.print(f"\n[bold green]>>> {label} - Basarili[/bold green]")
        return True
    else:
        console.print(
            f"\n[bold red]>>> {label} - Hata (kod: {result.returncode})[/bold red]"
        )
        return False


def display_results(language: str, config: dict) -> None:
    """Display evaluation results from results.json if available."""
    base_dir = config.get("output", {}).get("base_dir", "./output")
    results_path = Path(base_dir) / language / "results.json"

    if not results_path.exists():
        console.print("[yellow]Sonuc dosyasi bulunamadi.[/yellow]")
        return

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    table = Table(title="Degerlendirme Sonuclari", show_header=True, header_style="bold cyan")
    table.add_column("Metrik")
    table.add_column("Fine-tuned", justify="right", style="green")

    has_baseline = "baseline" in results and results["baseline"]
    if has_baseline:
        table.add_column("Baseline", justify="right", style="yellow")
        table.add_column("Iyilesme", justify="right", style="bold cyan")

    # WER
    row = ["WER", f"{results['wer']:.2%}"]
    if has_baseline:
        row.append(f"{results['baseline']['wer']:.2%}")
        imp = results.get("improvement", {}).get("wer")
        row.append(f"{imp:+.2%}" if imp is not None else "-")
    table.add_row(*row)

    # CER
    row = ["CER", f"{results['cer']:.2%}"]
    if has_baseline:
        row.append(f"{results['baseline']['cer']:.2%}")
        imp = results.get("improvement", {}).get("cer")
        row.append(f"{imp:+.2%}" if imp is not None else "-")
    table.add_row(*row)

    # RTFx
    row = ["RTFx", f"{results['rtfx']:.1f}x"]
    if has_baseline:
        row.append(f"{results['baseline']['rtfx']:.1f}x")
        row.append("-")
    table.add_row(*row)

    # Additional info
    table.add_row("Ornek Sayisi", str(results.get("num_samples", "-")))
    if has_baseline:
        table.add_row("", "", "")

    console.print(table)


# ---------------------------------------------------------------------------
# Pipeline steps
# ---------------------------------------------------------------------------

def step_prepare_data(config: dict) -> None:
    """Step 1: Data preparation."""
    console.rule("[bold]Adim 1: Veri Hazirlama[/bold]")
    lang = select_language()
    if not lang:
        return

    # Show relevant config
    console.print("\n[bold]Veri ayarlari:[/bold]")
    for key in ("sample_rate", "max_duration_seconds"):
        val = config.get("data", {}).get(key)
        if val is not None:
            console.print(f"  {key}: {val}")

    skip_augment = not Confirm.ask("Augmentation uygulansIn mI?", default=True)

    cmd = ["prepare_data.py", "--language", lang, "--config", "config.yaml"]
    if skip_augment:
        cmd.append("--no-augment")

    run_step("Veri Hazirlama", cmd)


def step_train_lora(config: dict) -> None:
    """Step 2: LoRA training."""
    console.rule("[bold]Adim 2: LoRA Egitimi[/bold]")
    lang = select_language()
    if not lang:
        return

    # Show training config
    console.print("\n[bold]Egitim ayarlari:[/bold]")
    for key, val in config.get("training", {}).items():
        console.print(f"  {key}: {val}")
    console.print("\n[bold]LoRA ayarlari:[/bold]")
    for key, val in config.get("lora", {}).items():
        console.print(f"  {key}: {val}")

    # Check for existing checkpoints
    base_dir = config.get("output", {}).get("base_dir", "./output")
    ckpt_dir = Path(base_dir) / lang / "checkpoints"
    cmd = ["train_lora.py", "--language", lang, "--config", "config.yaml"]

    if ckpt_dir.exists():
        checkpoints = sorted(ckpt_dir.glob("checkpoint-*"))
        if checkpoints:
            console.print(f"\n[yellow]Mevcut checkpoint'lar: {len(checkpoints)}[/yellow]")
            for cp in checkpoints[-3:]:
                console.print(f"  {cp.name}")
            if Confirm.ask("Son checkpoint'tan devam edilsin mi?", default=False):
                cmd.extend(["--resume-from-checkpoint", str(checkpoints[-1])])

    if not Confirm.ask("\nEgitimi baslat?", default=True):
        return

    run_step("LoRA Egitimi", cmd)


def step_merge_convert(config: dict) -> None:
    """Step 3: Merge LoRA weights and convert to CTranslate2."""
    console.rule("[bold]Adim 3: Birlestir & Donustur[/bold]")
    lang = select_language()
    if not lang:
        return

    quant = config.get("output", {}).get("ct2_quantization", "float16")
    console.print(f"\n[bold]CTranslate2 quantization:[/bold] {quant}")

    cmd = ["merge_and_convert.py", "--language", lang, "--config", "config.yaml"]
    run_step("Birlestir & Donustur", cmd)


def step_evaluate(config: dict) -> None:
    """Step 4: Evaluate the fine-tuned model."""
    console.rule("[bold]Adim 4: Degerlendirme[/bold]")
    lang = select_language()
    if not lang:
        return

    compare = Confirm.ask("Baseline (large-v3) ile karsilastirilsin mi?", default=True)

    cmd = ["evaluate.py", "--language", lang, "--config", "config.yaml"]
    if compare:
        cmd.append("--compare-baseline")

    success = run_step("Degerlendirme", cmd)
    if success:
        display_results(lang, config)


def step_full_pipeline(config: dict) -> None:
    """Step 5: Run full pipeline (all 4 steps sequentially)."""
    console.rule("[bold]Tam Pipeline[/bold]")
    lang = select_language()
    if not lang:
        return

    skip_augment = not Confirm.ask("Augmentation uygulansIn mI?", default=True)
    compare = Confirm.ask("Baseline karsilastirmasi yapilsin mi?", default=True)

    console.print(f"\n[bold]Dil:[/bold] {LANGUAGE_NAMES.get(lang, lang)} ({lang})")
    display_config(config)

    if not Confirm.ask("\nTum pipeline baslatilsin mi?", default=True):
        return

    # Step 1: Prepare data
    cmd1 = ["prepare_data.py", "--language", lang, "--config", "config.yaml"]
    if skip_augment:
        cmd1.append("--no-augment")
    if not run_step("1/4 Veri Hazirlama", cmd1):
        return

    # Step 2: Train
    cmd2 = ["train_lora.py", "--language", lang, "--config", "config.yaml"]
    if not run_step("2/4 LoRA Egitimi", cmd2):
        return

    # Step 3: Merge & Convert
    cmd3 = ["merge_and_convert.py", "--language", lang, "--config", "config.yaml"]
    if not run_step("3/4 Birlestir & Donustur", cmd3):
        return

    # Step 4: Evaluate
    cmd4 = ["evaluate.py", "--language", lang, "--config", "config.yaml"]
    if compare:
        cmd4.append("--compare-baseline")
    success = run_step("4/4 Degerlendirme", cmd4)

    if success:
        display_results(lang, config)
        base_dir = config.get("output", {}).get("base_dir", "./output")
        model_path = Path(base_dir) / lang / f"faster-whisper-{lang}"
        console.print(
            f"\n[bold green]Pipeline tamamlandi![/bold green]"
            f"\nModel: [bold]{model_path}[/bold]"
        )


def step_show_config(config: dict) -> None:
    """Step 6: Display current configuration."""
    console.rule("[bold]Yapilandirma[/bold]")
    display_config(config)


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

MENU_ITEMS = [
    ("1", "Veri Hazirla", step_prepare_data),
    ("2", "LoRA Egitimi", step_train_lora),
    ("3", "Birlestir & Donustur", step_merge_convert),
    ("4", "Degerlendir", step_evaluate),
    ("5", "Tum Pipeline'i Calistir", step_full_pipeline),
    ("6", "Yapilandirmayi Goster", step_show_config),
    ("0", "Cikis", None),
]


def show_menu() -> None:
    """Display the main menu panel."""
    lines = []
    for key, label, _ in MENU_ITEMS:
        if key == "0":
            lines.append("")
        lines.append(f"  [bold cyan][{key}][/bold cyan]  {label}")

    menu_text = "\n".join(lines)
    panel = Panel(
        menu_text,
        title="[bold]Whisper LoRA Fine-tuning[/bold]",
        border_style="blue",
        padding=(1, 2),
    )
    console.print(panel)


def main() -> None:
    """Main CLI entry point."""
    console.print(
        "\n[bold]Whisper LoRA Fine-tuning CLI[/bold]",
        style="blue",
    )
    console.print("[dim]Interaktif pipeline yonetimi. Ctrl+C ile iptal.[/dim]\n")

    config = load_config()

    dispatch = {key: func for key, _, func in MENU_ITEMS}

    while True:
        show_menu()
        choice = Prompt.ask("[bold]Seciminiz[/bold]", choices=[k for k, _, _ in MENU_ITEMS])

        if choice == "0":
            console.print("[dim]Cikis.[/dim]")
            break

        handler = dispatch.get(choice)
        if handler:
            handler(config)
        console.print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[dim]Iptal edildi.[/dim]")
        sys.exit(0)
