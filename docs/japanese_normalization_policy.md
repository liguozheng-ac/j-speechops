# Japanese Normalization Policy

## Policy

`japanese-text-prep-v1.1` defines `normalized_text` as a stable written form for downstream Japanese speech processing. It is not a reading, translation, paraphrase, or phoneme representation.

Processing order:

1. `jaconv.normalize(text, "NFKC")` converts compatibility forms, including half-width Katakana and full-width Latin letters/digits, and applies jaconv's documented punctuation mappings.
2. Commas are removed only from complete thousands-grouped integer tokens matching `1–3 digits + one or more comma-separated 3-digit groups`. Digit, comma, and decimal-point boundaries prevent partial matches inside malformed grouping or decimal tokens.
3. A long-vowel mark between Latin letters/digits is converted to ASCII `-`. This corrects the jaconv result for inputs such as `Ｗｉ－Ｆｉ` without changing Katakana long vowels.
4. Unicode whitespace runs collapse to one ASCII space; leading and trailing whitespace is removed.

Every applied category is recorded in `NormalizationResult.transformations`.

## Examples

```text
３,５００円        -> 3500円
1,234,567円       -> 1234567円
ﾁｪｯｸｲﾝ           -> チェックイン
Ｗｉ－Ｆｉ         -> Wi-Fi
日本語   English  -> 日本語 English
```

Malformed grouping (`12,34円`), decimals (`3.5%` and `1,234.5`), and ordinary punctuation (`こんにちは,世界` and `A,B`) are retained. Emoji and ordinary Japanese characters are retained. `raw_text` is never changed.

## What does not change

Normalization does not implement a general number engine, expand number words, infer counters, generate Kana, select pronunciation, alter meaning, run morphological analysis, or populate phonemes. In particular:

```text
normalized_text: 3500円
reading_kana:    サン…エン  # generated only by the reading layer
```

Therefore `normalized_text != reading_kana`. Context-sensitive readings for dates, time, amounts, people, and brands belong to the provider/override layer and may remain ambiguous.
