import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))

from pixelate_router.draft_audit import find_unresolved_markers


def _find_latex_main() -> Path:
    package_root = Path(__file__).resolve().parents[1]
    candidates = [
        package_root.parent / "deliverables" / "paper34_submission" / "latex_source" / "main.tex",
        package_root.parent / "latex_source" / "main.tex",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    pytest.skip("LaTeX source is not included in the standalone reproduction package.", allow_module_level=True)


LATEX_MAIN = _find_latex_main()


def test_finds_bracketed_numeric_markers():
    text = "Accuracy improves by [D1] and clean accuracy is [ACC_CLEAN]%."

    assert find_unresolved_markers(text) == ["[D1]", "[ACC_CLEAN]"]


def test_ignores_citation_numbers():
    text = "CIFAR-10-C was introduced in prior work [2]."

    assert find_unresolved_markers(text) == []


def test_manuscript_uses_pixelate_first_framing():
    text = LATEX_MAIN.read_text(encoding="utf-8")
    abstract = text.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]

    assert "pixelate" in abstract.lower()
    assert "5.18" in abstract
    assert "3.81" in abstract
    assert "1.39" in abstract
    assert abstract.index("5.18") < abstract.index("1.39")


def test_manuscript_avoids_broad_router_framing():
    text = LATEX_MAIN.read_text(encoding="utf-8").lower()
    forbidden = [
        "adaptive preprocessing",
        "contextual bandit",
        "contextual-bandit",
        "common image corruptions creates exploitable",
        "frequency-dependent jpeg residual bound predicts",
        "universal corruption",
        "general corruption router",
    ]

    for phrase in forbidden:
        assert phrase not in text


def test_manuscript_uses_compact_no_placeholder_tables():
    text = LATEX_MAIN.read_text(encoding="utf-8")
    inputs = [line for line in text.splitlines() if r"\input{tables/imagenetc/table_" in line]

    assert len(inputs) <= 4
    assert any("table_pixelate_primary.tex" in line for line in inputs)
    assert any("table_main.tex" in line for line in inputs)
    assert any("table_ablation.tex" in line for line in inputs)
    assert not any("table_oracle.tex" in line for line in inputs)
    assert not any("table_timing.tex" in line for line in inputs)
    assert not any("table_action_distribution.tex" in line for line in inputs)

    tables_dir = LATEX_MAIN.parent / "tables" / "imagenetc"
    combined = "\n".join(
        (tables_dir / name).read_text(encoding="utf-8")
        for name in ["table_pixelate_primary.tex", "table_main.tex", "table_ablation.tex"]
    )
    assert "--" not in combined
    assert " &  & " not in combined


def test_manuscript_uses_redesigned_figures():
    text = LATEX_MAIN.read_text(encoding="utf-8")

    assert r"\begin{figure*}" not in text
    assert "figure_case_mechanism.pdf" in text
    assert "figure_residual_routing.pdf" in text
    assert "figure_pixelate_severity.pdf" in text
    assert "figure_pipeline.pdf" not in text
    assert "figure_radial_spectra.pdf" not in text
    assert "figure_severity_curves.pdf" not in text


def test_reproducibility_note_is_inline():
    text = LATEX_MAIN.read_text(encoding="utf-8")

    assert r"\subsection{Reproducibility}" not in text
    assert r"\section*{Data and Code Availability}" not in text
    assert "https://github.com/AdML123/pixelate-action-selection" in text
    assert "not redistributed" in text


def test_manuscript_avoids_ai_like_prose_markers():
    text = LATEX_MAIN.read_text(encoding="utf-8")
    lower = text.lower()
    forbidden = [
        "\u2014",
        "delve into",
        "shed light on",
        "advance understanding",
        "paves the way",
        "opens new avenues",
        "taken together",
        "overall,",
        "not only",
        "but also",
        "rather than",
        "in contrast",
        "while this study",
        "this revision",
        "first,",
        "second,",
        "third,",
    ]

    for phrase in forbidden:
        assert phrase not in lower


def test_body_avoids_template_like_scaffolding():
    text = LATEX_MAIN.read_text(encoding="utf-8")
    body = text.split(r"\begin{thebibliography}", 1)[0].lower()
    forbidden = [
        "this paper",
        "the paper",
        "this study",
        "the contribution is",
        "the result is",
        "the method is",
        "the router uses this",
        "overall",
        "taken together",
        "delve into",
        "shed light on",
        "advance understanding",
    ]

    for phrase in forbidden:
        assert phrase not in body

    assert body.count(";") <= 1
    assert "\u2014" not in body


def test_first_citation_order_is_monotonic():
    text = LATEX_MAIN.read_text(encoding="utf-8")
    body = text.split(r"\begin{thebibliography}", 1)[0]
    bibliography = text.split(r"\begin{thebibliography}", 1)[1]
    keys = re.findall(r"\\bibitem\{([^}]+)\}", bibliography)
    order = {key: index + 1 for index, key in enumerate(keys)}
    seen: list[int] = []
    for citation in re.findall(r"\\cite\{([^}]+)\}", body):
        for key in [item.strip() for item in citation.split(",")]:
            seen.append(order[key])

    assert seen == sorted(seen)


def test_abstract_and_keywords_follow_ieee_shape():
    text = LATEX_MAIN.read_text(encoding="utf-8")
    abstract = text.split(r"\begin{abstract}", 1)[1].split(r"\end{abstract}", 1)[0]
    keywords = text.split(r"\begin{IEEEkeywords}", 1)[1].split(r"\end{IEEEkeywords}", 1)[0]

    assert len(re.findall(r"\b\w+\b", abstract)) <= 250
    assert r"\cite" not in abstract
    assert r"\begin{equation}" not in abstract
    assert "$" not in abstract
    keyword_items = [item.strip() for item in keywords.replace("\n", " ").split(",") if item.strip()]
    assert 3 <= len(keyword_items) <= 5


def test_required_abbreviations_and_symbols_are_defined():
    text = LATEX_MAIN.read_text(encoding="utf-8")
    lower = text.lower()
    required = [
        "deep convolutional denoising network",
        "residual network (resnet)-50",
        "joint photographic experts group (jpeg)",
        "imagenet-c",
        "leave-one-corruption-out",
        "lower confidence bound",
    ]

    for phrase in required:
        assert phrase in lower

    for symbol in ["r_J(x)", "J_{20}", "Q_\\theta", "\\Delta Q", "\\tau"]:
        assert symbol in text


def test_camera_ready_method_and_baseline_details_are_present():
    text = LATEX_MAIN.read_text(encoding="utf-8")
    required = [
        "Config-A is $D(J_{20}(x))$",
        "Config-B is $J_{20}(D(x))$",
        "Config-A10 is $D(J_{10}(x))$",
        "19-dimensional feature vector",
        "at or above 0.25",
        "below 0.15",
        "from 0.15 to below 0.30",
        "50 epochs",
        "batch size 256",
        "learning rate 0.001",
        "weight decay 0.0001",
        "early-stopping patience 8",
        "seed 3407",
        "0.00, 0.01, 0.02, 0.03, 0.05, 0.10, and 0.15",
        "HFER rules",
        "no-commutator router",
        "corruption-detector policy",
        "72.5 percent detector accuracy",
        "65.2 ms median overhead",
        "dncnn\\_25.pth",
        "grayscale",
        "RGB blind DnCNN action space",
    ]

    for phrase in required:
        assert phrase in text


def test_generated_tables_avoid_old_bandit_labels():
    tables_dir = LATEX_MAIN.parent / "tables" / "imagenetc"
    combined = "\n".join(path.read_text(encoding="utf-8") for path in tables_dir.glob("table_*.tex"))
    forbidden = ["Bandit", "Logistic CB", "CB:"]

    for phrase in forbidden:
        assert phrase not in combined
