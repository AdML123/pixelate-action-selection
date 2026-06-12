import re

_MARKER = re.compile(r"\[[A-Z][A-Z0-9_{}\\^$-]*\]")
_LATEX_OPTION_PREFIX = re.compile(r"\\[A-Za-z]+\s*$")


def find_unresolved_markers(text: str) -> list[str]:
    markers = []
    for match in _MARKER.finditer(text):
        prefix = text[max(0, match.start() - 40) : match.start()]
        if _LATEX_OPTION_PREFIX.search(prefix):
            continue
        markers.append(match.group(0))
    return markers
