import json
import os
import re
import unicodedata
from fractions import Fraction
from typing import Callable, Dict, Iterator, List, Optional, Union

import regex
from more_itertools import windowed

def to_fraction(value: str) -> Fraction:
    try:
        return Fraction(value)
    except ValueError:
        print(f"Could not convert {value} to a fraction")
        return None
# Constants for diacritics replacement
ADDITIONAL_DIACRITICS: Dict[str, str] = {
    "œ": "oe",
    "Œ": "OE",
    "ø": "o",
    "Ø": "O",
    "æ": "ae",
    "Æ": "AE",
    "ß": "ss",
    "ẞ": "SS",
    "đ": "d",
    "Đ": "D",
    "ð": "d",
    "Ð": "D",
    "þ": "th",
    "Þ": "th",
    "ł": "l",
    "Ł": "L",
}


def remove_symbols_and_diacritics(s: str, keep: str = "") -> str:
    """
    Replace symbols and punctuation with a space,
    drop diacritics, and apply additional diacritics mappings.

    Args:
        s (str): Input string.
        keep (str): Characters to retain as-is.

    Returns:
        str: Normalized string.
    """
    try:
        normalized = unicodedata.normalize("NFKD", s)
    except TypeError as e:
        raise ValueError(f"Input must be a string, got {type(s)}") from e

    result = []
    for c in normalized:
        if c in keep:
            result.append(c)
        elif c in ADDITIONAL_DIACRITICS:
            result.append(ADDITIONAL_DIACRITICS[c])
        elif unicodedata.category(c) == "Mn":
            continue
        elif unicodedata.category(c)[0] in {"M", "S", "P"}:
            result.append(" ")
        else:
            result.append(c)
    return "".join(result)


def remove_symbols(s: str) -> str:
    """
    Replace symbols and punctuation with a space while keeping diacritics.

    Args:
        s (str): Input string.

    Returns:
        str: String with symbols removed.
    """
    try:
        normalized = unicodedata.normalize("NFKC", s)
    except TypeError as e:
        raise ValueError(f"Input must be a string, got {type(s)}") from e

    return "".join(" " if unicodedata.category(c)[0] in {"M", "S", "P"} else c for c in normalized)


class BasicTextNormalizer:
    """
    A basic text normalizer that can remove symbols and diacritics,
    with options to split letters and customize behavior.
    """

    def __init__(self, remove_diacritics: bool = False, split_letters: bool = False) -> None:
        """
        Initialize the BasicTextNormalizer.

        Args:
            remove_diacritics (bool): Whether to remove diacritics.
            split_letters (bool): Whether to split letters into separate tokens.
        """
        self.clean: Callable[[str], str] = (
            remove_symbols_and_diacritics if remove_diacritics else remove_symbols
        )
        self.split_letters = split_letters

    def __call__(self, s: str) -> str:
        """
        Normalize the input string.

        Args:
            s (str): Input string.

        Returns:
            str: Normalized string.
        """
        if not isinstance(s, str):
            raise TypeError("Input must be a string.")

        s = s.lower()
        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # Remove content within brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # Remove content within parentheses
        s = self.clean(s).lower()

        if self.split_letters:
            s = " ".join(regex.findall(r"\X", s, regex.U))

        s = re.sub(r"\s+", " ", s)  # Replace multiple whitespaces with a single space

        return s.strip()


class EnglishNumberNormalizer:
    """
    Normalize English spelled-out numbers into Arabic numerals.
    Handles various cases including ordinals, plurals, currencies, etc.
    """

    def __init__(self) -> None:
        super().__init__()
        self.zeros = {"o", "oh", "zero"}
        self.ones: Dict[str, int] = {
            name: i
            for i, name in enumerate(
                [
                    "one",
                    "two",
                    "three",
                    "four",
                    "five",
                    "six",
                    "seven",
                    "eight",
                    "nine",
                    "ten",
                    "eleven",
                    "twelve",
                    "thirteen",
                    "fourteen",
                    "fifteen",
                    "sixteen",
                    "seventeen",
                    "eighteen",
                    "nineteen",
                ],
                start=1,
            )
        }
        self.ones_plural: Dict[str, tuple[int, str]] = {
            "sixes" if name == "six" else name + "s": (value, "s")
            for name, value in self.ones.items()
        }
        self.ones_ordinal: Dict[str, tuple[int, str]] = {
            "zeroth": (0, "th"),
            "first": (1, "st"),
            "second": (2, "nd"),
            "third": (3, "rd"),
            "fifth": (5, "th"),
            "twelfth": (12, "th"),
            **{
                name + ("h" if name.endswith("t") else "th"): (value, "th")
                for name, value in self.ones.items()
                if value > 3 and value != 5 and value != 12
            },
        }
        self.ones_suffixed: Dict[str, tuple[int, str]] = {**self.ones_plural, **self.ones_ordinal}

        self.tens: Dict[str, int] = {
            "twenty": 20,
            "thirty": 30,
            "forty": 40,
            "fifty": 50,
            "sixty": 60,
            "seventy": 70,
            "eighty": 80,
            "ninety": 90,
        }
        self.tens_plural: Dict[str, tuple[int, str]] = {
            name.replace("y", "ies"): (value, "s") for name, value in self.tens.items()
        }
        self.tens_ordinal: Dict[str, tuple[int, str]] = {
            name.replace("y", "ieth"): (value, "th") for name, value in self.tens.items()
        }
        self.tens_suffixed: Dict[str, tuple[int, str]] = {**self.tens_plural, **self.tens_ordinal}

        self.multipliers: Dict[str, int] = {
            "hundred": 100,
            "thousand": 1_000,
            "million": 1_000_000,
            "billion": 1_000_000_000,
            "trillion": 1_000_000_000_000,
            "quadrillion": 1_000_000_000_000_000,
            "quintillion": 1_000_000_000_000_000_000,
            "sextillion": 1_000_000_000_000_000_000_000,
            "septillion": 1_000_000_000_000_000_000_000_000,
            "octillion": 1_000_000_000_000_000_000_000_000_000,
            "nonillion": 1_000_000_000_000_000_000_000_000_000_000,
            "decillion": 1_000_000_000_000_000_000_000_000_000_000_000,
        }
        self.multipliers_plural: Dict[str, tuple[int, str]] = {
            name + "s": (value, "s") for name, value in self.multipliers.items()
        }
        self.multipliers_ordinal: Dict[str, tuple[int, str]] = {
            name + "th": (value, "th") for name, value in self.multipliers.items()
        }
        self.multipliers_suffixed: Dict[str, tuple[int, str]] = {
            **self.multipliers_plural,
            **self.multipliers_ordinal,
        }

        self.decimals: set = {*self.ones, *self.tens, *self.zeros}

        self.preceding_prefixers: Dict[str, str] = {
            "minus": "-",
            "negative": "-",
            "plus": "+",
            "positive": "+",
        }
        self.following_prefixers: Dict[str, str] = {
            "pound": "£",
            "pounds": "£",
            "euro": "€",
            "euros": "€",
            "dollar": "$",
            "dollars": "$",
            "cent": "¢",
            "cents": "¢",
        }
        self.prefixes: set = set(
            list(self.preceding_prefixers.values()) + list(self.following_prefixers.values())
        )
        self.suffixers: Dict[str, Union[str, Dict[str, str]]] = {
            "per": {"cent": "%"},
            "percent": "%",
        }
        self.specials: set = {"and", "double", "triple", "point"}

        self.words: set = {
            key
            for mapping in [
                self.zeros,
                self.ones,
                self.ones_suffixed,
                self.tens,
                self.tens_suffixed,
                self.multipliers,
                self.multipliers_suffixed,
                self.preceding_prefixers,
                self.following_prefixers,
                self.suffixers,
                self.specials,
            ]
            for key in mapping
        }
        self.literal_words: set = {"one", "ones"}

    def process_words(self, words: List[str]) -> Iterator[str]:
        """
        Process a list of words and yield normalized tokens.

        Args:
            words (List[str]): List of words to process.

        Yields:
            Iterator[str]: Normalized tokens.
        """
        prefix: Optional[str] = None
        value: Optional[Union[str, int]] = None
        skip: bool = False

        def to_fraction(s: str) -> Optional[Fraction]:
            try:
                return Fraction(s)
            except ValueError:
                return None

        def output(result: Union[str, int]) -> str:
            nonlocal prefix, value
            result_str = str(result)
            if prefix:
                result_str = prefix + result_str
            prefix = None
            value = None
            return result_str

        if not words:
            return

        for prev, current, next_ in windowed([None] + words + [None], 3):
            if skip:
                skip = False
                continue

            if current is None:
                continue

            next_is_numeric: bool = next_ is not None and re.match(r"^\d+(\.\d+)?$", next_)
            has_prefix: bool = current[0] in self.prefixes if current else False
            current_without_prefix: str = current[1:] if has_prefix else current

            if re.match(r"^\d+(\.\d+)?$", current_without_prefix):
                f = to_fraction(current_without_prefix)
                if f is None:
                    yield from self._handle_non_numeric(current, prev, value, prefix)
                    continue

                if value is not None:
                    if isinstance(value, str) and value.endswith("."):
                        value = f"{value}{current}"
                        continue
                    else:
                        yield output(value)

                prefix = current[0] if has_prefix else prefix
                value = f.numerator if f.denominator == 1 else current_without_prefix
            elif current not in self.words:
                if value is not None:
                    yield output(value)
                yield output(current)
            elif current in self.zeros:
                value = f"{value or ''}0"
            elif current in self.ones:
                ones = self.ones[current]
                value = self._handle_ones(current, prev, value, ones)
            elif current in self.ones_suffixed:
                yield from self._handle_ones_suffixed(current, prev, value, prefix)
                continue
            elif current in self.tens:
                tens = self.tens[current]
                value = self._handle_tens(tens, value)
            elif current in self.tens_suffixed:
                yield from self._handle_tens_suffixed(current, prev, value, prefix)
                continue
            elif current in self.multipliers:
                value = self._handle_multipliers(current, value)
            elif current in self.multipliers_suffixed:
                yield from self._handle_multipliers_suffixed(current, prev, value, prefix)
                continue
            elif current in self.preceding_prefixers:
                yield from self._handle_preceding_prefixers(current, next_, value, prefix)
            elif current in self.following_prefixers:
                yield from self._handle_following_prefixers(current, value, prefix)
            elif current in self.suffixers:
                yield from self._handle_suffixers(current, next_, value)
            elif current in self.specials:
                yield from self._handle_specials(current, next_, value)
            else:
                raise ValueError(f"Unexpected token: {current}")

        if value is not None:
            yield output(value)

    def preprocess(self, s: str) -> str:
        """
        Preprocess the input string by handling specific patterns.

        Args:
            s (str): Input string.

        Returns:
            str: Preprocessed string.
        """
        if not isinstance(s, str):
            raise TypeError(f"Expected string for preprocessing, got {type(s)}")

        results = []

        segments = re.split(r"\band\s+a\s+half\b", s)
        for i, segment in enumerate(segments):
            segment = segment.strip()
            if not segment:
                continue
            if i < len(segments) - 1:
                last_word = segment.rsplit(maxsplit=2)[-1] if len(segment.split()) >= 2 else segment
                if last_word in self.decimals or last_word in self.multipliers:
                    results.append(f"{segment} point five")
                else:
                    results.append(f"{segment} and a half")
            else:
                results.append(segment)

        s = " ".join(results)
        s = re.sub(r"([a-z])([0-9])", r"\1 \2", s)
        s = re.sub(r"([0-9])([a-z])", r"\1 \2", s)
        s = re.sub(r"([0-9])\s+(st|nd|rd|th|s)\b", r"\1\2", s)

        return s

    def postprocess(self, s: str) -> str:
        """
        Postprocess the normalized string to handle special cases.

        Args:
            s (str): Normalized string.

        Returns:
            str: Postprocessed string.
        """
        def combine_cents(m: re.Match) -> str:
            try:
                currency = m.group(1)
                integer = m.group(2)
                cents = int(m.group(3))
                return f"{currency}{integer}.{cents:02d}"
            except (ValueError, IndexError):
                return m.group(0)

        def extract_cents(m: re.Match) -> str:
            try:
                return f"¢{int(m.group(1))}"
            except (ValueError, IndexError):
                return m.group(0)

        s = re.sub(r"([€£$])([0-9]+) (?:and )?¢([0-9]{1,2})\b", combine_cents, s)
        s = re.sub(r"[€£$]0\.([0-9]{1,2})\b", extract_cents, s)
        s = re.sub(r"\b1(s?)\b", r"one\1", s)

        return s

    def __call__(self, s: str) -> str:
        """
        Normalize the input string by preprocessing, processing words, and postprocessing.

        Args:
            s (str): Input string.

        Returns:
            str: Fully normalized string.
        """
        if not isinstance(s, str):
            raise TypeError("Input must be a string.")

        s = self.preprocess(s)
        words = s.split()
        normalized_words = [word for word in self.process_words(words) if word]
        s = " ".join(normalized_words)
        s = self.postprocess(s)

        s = re.sub(r"\s+", " ", s)  # Replace any successive whitespace with a space

        return s.strip()

    # Helper methods for processing different parts
    def _handle_ones(self, current: str, prev: Optional[str], value: Optional[Union[str, int]], ones: int) -> Union[str, int]:
        if value is None:
            return ones
        if isinstance(value, str) or prev in self.ones:
            if prev in self.tens and ones < 10:
                if isinstance(value, str) and value.endswith("0"):
                    return value[:-1] + str(ones)
            return f"{value}{ones}"
        if ones < 10:
            if isinstance(value, int) and value % 10 == 0:
                return value + ones
            return f"{value}{ones}"
        else:
            if isinstance(value, int) and value % 100 == 0:
                return value + ones
            return f"{value}{ones}"

    def _handle_ones_suffixed(self, current: str, prev: Optional[str], value: Optional[Union[str, int]], prefix: Optional[str]) -> Iterator[str]:
        ones, suffix = self.ones_suffixed[current]
        if value is None:
            yield self._output_with_prefix(ones, suffix, prefix)
        elif isinstance(value, str) or (prev and prev in self.ones):
            if prev in self.tens and ones < 10:
                if isinstance(value, str) and value.endswith("0"):
                    yield self._output_with_prefix(value[:-1] + str(ones), suffix, prefix)
                    return
            else:
                yield self._output_with_prefix(f"{value}{ones}", suffix, prefix)
        elif isinstance(value, int):
            if ones < 10:
                if value % 10 == 0:
                    yield self._output_with_prefix(value + ones, suffix, prefix)
                else:
                    yield self._output_with_prefix(f"{value}{ones}", suffix, prefix)
            else:
                if value % 100 == 0:
                    yield self._output_with_prefix(value + ones, suffix, prefix)
                else:
                    yield self._output_with_prefix(f"{value}{ones}", suffix, prefix)
        value = None
        prefix = None

    def _handle_tens(self, tens: int, value: Optional[Union[str, int]]) -> Union[str, int]:
        if value is None:
            return tens
        if isinstance(value, str):
            return f"{value}{tens}"
        if isinstance(value, int):
            if value % 100 == 0:
                return value + tens
            return f"{value}{tens}"
        return value

    def _handle_tens_suffixed(self, current: str, prev: Optional[str], value: Optional[Union[str, int]], prefix: Optional[str]) -> Iterator[str]:
        tens, suffix = self.tens_suffixed[current]
        if value is None:
            yield self._output_with_prefix(tens, suffix, prefix)
        elif isinstance(value, str):
            yield self._output_with_prefix(f"{value}{tens}", suffix, prefix)
        elif isinstance(value, int):
            if value % 100 == 0:
                yield self._output_with_prefix(value + tens, suffix, prefix)
            else:
                yield self._output_with_prefix(f"{value}{tens}", suffix, prefix)
        value = None
        prefix = None

    def _handle_multipliers(self, current: str, value: Optional[Union[str, int]]) -> Union[str, int, None]:
        multiplier = self.multipliers[current]
        if value is None:
            return multiplier
        if isinstance(value, str) or value == 0:
            f = to_fraction(value) if isinstance(value, str) else None
            if f is not None:
                p = f * multiplier
                if p.denominator == 1:
                    return p.numerator
            return value
        elif isinstance(value, int):
            before = (value // 1000) * 1000
            residual = value % 1000
            return before + residual * multiplier
        return value

    def _handle_multipliers_suffixed(
        self, current: str, prev: Optional[str], value: Optional[Union[str, int]], prefix: Optional[str]
    ) -> Iterator[str]:
        multiplier, suffix = self.multipliers_suffixed[current]
        if value is None:
            yield self._output_with_prefix(multiplier, suffix, prefix)
        elif isinstance(value, str):
            f = to_fraction(value) if isinstance(value, str) else None
            if f is not None:
                p = f * multiplier
                if p.denominator == 1:
                    yield self._output_with_prefix(p.numerator, suffix, prefix)
                else:
                    yield self._output_with_prefix(value, "")
                    yield self._output_with_prefix(multiplier, suffix, prefix)
            else:
                yield self._output_with_prefix(value, "")
                yield self._output_with_prefix(multiplier, suffix, prefix)
        elif isinstance(value, int):
            before = (value // 1000) * 1000
            residual = value % 1000
            value = before + residual * multiplier
            yield self._output_with_prefix(value, suffix, prefix)
        value = None
        prefix = None

    def _handle_preceding_prefixers(
        self, current: str, next_: Optional[str], value: Optional[Union[str, int]], prefix: Optional[str]
    ) -> Iterator[str]:
        if value is not None:
            yield self._output_with_prefix(value, "", prefix)
        if next_ in self.words or (next_ and re.match(r"^\d+(\.\d+)?$", next_)):
            prefix = self.preceding_prefixers[current]
        else:
            yield current
        return

    def _handle_following_prefixers(
        self, current: str, value: Optional[Union[str, int]], prefix: Optional[str]
    ) -> Iterator[str]:
        if value is not None:
            prefix = self.following_prefixers[current]
            yield self._output_with_prefix(value, "", prefix)
        else:
            yield current
        return

    def _handle_suffixers(self, current: str, next_: Optional[str], value: Optional[Union[str, int]]) -> Iterator[str]:
        if value is not None:
            suffix = self.suffixers[current]
            if isinstance(suffix, dict):
                if next_ in suffix:
                    yield f"{value}{suffix[next_]}"
                else:
                    yield str(value)
                    yield current
            else:
                yield f"{value}{suffix}"
        else:
            yield current
        return

    def _handle_specials(self, current: str, next_: Optional[str], value: Optional[Union[str, int]]) -> Iterator[str]:
        if next_ not in self.words and not (next_ and re.match(r"^\d+(\.\d+)?$", next_)):
            if value is not None:
                yield str(value)
            yield current
        elif current == "and":
            if "hundred" not in self.words:
                if value is not None:
                    yield str(value)
                yield current
        elif current in {"double", "triple"} and next_ in self.ones.union(self.zeros):
            repeats = 2 if current == "double" else 3
            digit = self.ones.get(next_, "0")
            value = f"{value or ''}{str(digit) * repeats}"
            skip = True
        elif current == "point" and (next_ in self.decimals or (next_ and re.match(r"^\d+(\.\d+)?$", next_))):
            value = f"{value or ''}."
        else:
            yield current
        return

    def _output_with_prefix(self, value: Union[str, int], suffix: str, prefix: Optional[str]) -> str:
        """
        Helper method to apply prefix and suffix to a value.

        Args:
            value (Union[str, int]): The numeric value.
            suffix (str): Suffix to append.
            prefix (Optional[str]): Prefix to prepend.

        Returns:
            str: Formatted string with prefix and suffix.
        """
        result = str(value)
        if prefix:
            result = f"{prefix}{result}"
        if suffix:
            result += suffix
        return result


class EnglishSpellingNormalizer:
    """
    Applies British-American spelling mappings as listed in [1].

    [1] https://www.tysto.com/uk-us-spelling-list.html
    """

    def __init__(self):
        mapping_path = os.path.join(os.path.dirname(__file__), "english.json")
        self.mapping = json.load(open(mapping_path))

    def __call__(self, s: str):
        return " ".join(self.mapping.get(word, word) for word in s.split())


class EnglishTextNormalizer:
    def __init__(self):
        self.ignore_patterns = r"\b(hmm|mm|mhm|mmm|uh|um)\b"
        self.replacers = {
            # common contractions
            r"\bwon't\b": "will not",
            r"\bcan't\b": "can not",
            r"\blet's\b": "let us",
            r"\bain't\b": "aint",
            r"\by'all\b": "you all",
            r"\bwanna\b": "want to",
            r"\bgotta\b": "got to",
            r"\bgonna\b": "going to",
            r"\bi'ma\b": "i am going to",
            r"\bimma\b": "i am going to",
            r"\bwoulda\b": "would have",
            r"\bcoulda\b": "could have",
            r"\bshoulda\b": "should have",
            r"\bma'am\b": "madam",
            # contractions in titles/prefixes
            r"\bmr\b": "mister ",
            r"\bmrs\b": "missus ",
            r"\bst\b": "saint ",
            r"\bdr\b": "doctor ",
            r"\bprof\b": "professor ",
            r"\bcapt\b": "captain ",
            r"\bgov\b": "governor ",
            r"\bald\b": "alderman ",
            r"\bgen\b": "general ",
            r"\bsen\b": "senator ",
            r"\brep\b": "representative ",
            r"\bpres\b": "president ",
            r"\brev\b": "reverend ",
            r"\bhon\b": "honorable ",
            r"\basst\b": "assistant ",
            r"\bassoc\b": "associate ",
            r"\blt\b": "lieutenant ",
            r"\bcol\b": "colonel ",
            r"\bjr\b": "junior ",
            r"\bsr\b": "senior ",
            r"\besq\b": "esquire ",
            # prefect tenses, ideally it should be any past participles, but it's harder..
            r"'d been\b": " had been",
            r"'s been\b": " has been",
            r"'d gone\b": " had gone",
            r"'s gone\b": " has gone",
            r"'d done\b": " had done",  # "'s done" is ambiguous
            r"'s got\b": " has got",
            # general contractions
            r"n't\b": " not",
            r"'re\b": " are",
            r"'s\b": " is",
            r"'d\b": " would",
            r"'ll\b": " will",
            r"'t\b": " not",
            r"'ve\b": " have",
            r"'m\b": " am",
        }
        self.standardize_numbers = EnglishNumberNormalizer()
        self.standardize_spellings = EnglishSpellingNormalizer()

    def __call__(self, s: str):
        s = s.lower()

        s = re.sub(r"[<\[][^>\]]*[>\]]", "", s)  # remove words between brackets
        s = re.sub(r"\(([^)]+?)\)", "", s)  # remove words between parenthesis
        s = re.sub(self.ignore_patterns, "", s)
        s = re.sub(r"\s+'", "'", s)  # when there's a space before an apostrophe

        for pattern, replacement in self.replacers.items():
            s = re.sub(pattern, replacement, s)

        s = re.sub(r"(\d),(\d)", r"\1\2", s)  # remove commas between digits
        s = re.sub(r"\.([^0-9]|$)", r" \1", s)  # remove periods not followed by numbers
        s = remove_symbols_and_diacritics(s, keep=".%$¢€£")  # keep numeric symbols

        s = self.standardize_numbers(s)
        s = self.standardize_spellings(s)

        # now remove prefix/suffix symbols that are not preceded/followed by numbers
        s = re.sub(r"[.$¢€£]([^0-9])", r" \1", s)
        s = re.sub(r"([^0-9])%", r"\1 ", s)

        s = re.sub(r"\s+", " ", s)  # replace any successive whitespaces with a space

        return s

def main() -> None:
    text = "I can't believe it's already 2023! Let's meet at 5th Avenue."

    basic_normalizer = BasicTextNormalizer(remove_diacritics=True, split_letters=False)
    basic_normalized = basic_normalizer(text)
    print("Basic Normalized:", basic_normalized)

    number_normalizer = EnglishNumberNormalizer()
    number_normalized = number_normalizer(text)
    print("Number Normalized:", number_normalized)

    spelling_normalizer = EnglishSpellingNormalizer()
    spelling_normalized = spelling_normalizer(text)
    print("Spelling Normalized:", spelling_normalized)

    english_normalizer = EnglishTextNormalizer()
    english_normalized = english_normalizer(text)
    print("English Normalized:", english_normalized)

if __name__ == "__main__":
    main()