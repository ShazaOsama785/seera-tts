import re

_MISHKAL_MARKER = "\x01"  # mishkal swaps . ! : for this control char
_RESTORABLE = re.compile(r"[.!:]")


class MishkalDiacritizer:
    def __init__(self) -> None:
        try:
            import mishkal.tashkeel
        except ImportError as exc:
            raise ImportError("Install with: pip install mishkal") from exc
        self._vocalizer = mishkal.tashkeel.TashkeelClass()

    def __call__(self, text: str) -> str:
        return _clean_mishkal(text, self._vocalizer.tashkeel(text))


def _clean_mishkal(original: str, output: str) -> str:
    """Put back the punctuation mishkal removed and tidy its spacing."""
    puncts = iter(_RESTORABLE.findall(original))
    output = output.replace(_MISHKAL_MARKER, "\x00")
    output = re.sub("\x00", lambda _: next(puncts, "."), output)
    output = re.sub(r"\s+([،,.؛:!؟?])", r"\1", output)   # no space before punctuation
    output = re.sub(r"([.!:])(?=\S)", r"\1 ", output)     # one space after it
    return re.sub(r"\s+", " ", output).strip()