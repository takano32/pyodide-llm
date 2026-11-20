# tests/wllama_compare.py's two pure parts (T519 review): how far two answers begin alike, and the table the results turn
# into. The rest of the tool needs a network and a browser (CI's).
import wllama_compare as w

BASE = dict(os="linux x64", cpu="a cpu", cores=4, threads=2, browserVersion="1", wllama="3.13.0", query="")


def lines(page, theirs, page_text="Hello world", their_text="Hello world"):
    out = []
    for side, (prompt, answer), text in (("page", page, page_text), ("wllama", theirs, their_text)):
        out.append(dict(BASE, model="m", side=side, what="load", fetched=True, seconds=10.0))
        out.append(dict(BASE, model="m", side=side, what="load", fetched=False, seconds=4.0))
        # a warm-up answer (what="warm") is not a round: the table leaves it out
        out.append(dict(BASE, model="m", side=side, what="warm", promptTokens=1, promptPerSecond=1.0, answerTokens=1, answerPerSecond=1.0, text="x"))
        for round_ in range(3):
            out.append(dict(BASE, model="m", side=side, what="answer", round=round_, promptTokens=45, promptPerSecond=prompt,
                            answerTokens=131, answerPerSecond=answer + round_ * 0.1, text=text))
    return out


def test_same_start_counts_words_not_markdown():
    # the page draws the Markdown (no "**", one line where the text had a blank one), wllama's text is as written
    assert w.same_start("Hello **world**\n\nfoo", "Hello world foo bar") == "15 of 15"
    assert w.same_start("Hello world", "Hello there") == "6 of 11"
    assert w.same_start("", "abc") == "0 of 0"


def test_table_gives_the_medians_and_the_ratios():
    table = w.table(lines((90.0, 40.0), (10.0, 8.0)))
    page, theirs = [row for row in table.splitlines() if row.startswith("| m |")]
    # the median of 40.0, 40.1, 40.2 and the ratios page ÷ wllama (the warm-up's 1.0 is not in them)
    assert "| 40.1 (40.0–40.2) |" in page
    assert "| 9.00 | 4.95 |" in page
    assert "| 8.1 (8.0–8.2) |" in theirs and theirs.endswith("|  |  |  |")
    assert "4 logical cores, 2 threads each; Chromium 1, wllama 3.13.0" in table


def test_table_names_what_failed_and_the_page_s_query():
    rows = lines((90.0, 40.0), (10.0, 8.0)) + [dict(BASE, model="m", side="tool", what="failures", failures=["wllama ran on one thread"])]
    assert "- m: wllama ran on one thread" in w.table(rows)
    assert "the page with ?without=relaxed" in w.table([dict(row, query="without=relaxed") for row in lines((90.0, 40.0), (10.0, 8.0))])
