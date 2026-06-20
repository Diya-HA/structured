#!/usr/bin/env python3
"""
extract.py — Extract structured data from an unstructured document.

What it does
------------
1. Reads an input file (HTML, Markdown, or plain text).
2. Detects the type from the file extension.
3. If the file is HTML, converts it to Markdown first, trying to keep as
   much of the content as possible (tables, links, lists, headings).
4. Sends the text to Anthropic's Claude API along with a JSON schema you
   provide, and asks Claude to return data that matches that schema.
5. Prints the result as JSON (or writes it to a file with --output).

How it forces the output to match your schema
---------------------------------------------
The script hands Claude a "tool" whose input schema is exactly the JSON
schema you supply, and tells Claude it must call that tool. Claude then
fills in the tool's arguments, which is the cleanest way to get output
that conforms to a schema.

Usage
-----
    export ANTHROPIC_API_KEY="sk-ant-..."
    python extract.py --input document.html --schema schema.json
    python extract.py --input notes.md --schema schema.json --output result.json

Install dependencies
--------------------
    pip install anthropic
    # Optional, for the best HTML -> Markdown conversion:
    pip install markdownify
    # (html2text is used as a fallback if markdownify isn't installed)
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Extensions we recognize, grouped by how we treat them.
HTML_EXTS = {".html", ".htm", ".xhtml"}
MARKDOWN_EXTS = {".md", ".markdown", ".mdown", ".mkd"}
TEXT_EXTS = {".txt", ".text", ""}

DEFAULT_MODEL = "claude-sonnet-4-6"


def detect_type(path: Path) -> str:
    """Return 'html', 'markdown', or 'text' based on the file extension."""
    ext = path.suffix.lower()
    if ext in HTML_EXTS:
        return "html"
    if ext in MARKDOWN_EXTS:
        return "markdown"
    if ext in TEXT_EXTS:
        return "text"
    # Unknown extension: treat it as plain text rather than failing.
    print(
        f"Warning: unrecognized extension '{ext}', treating file as plain text.",
        file=sys.stderr,
    )
    return "text"


def html_to_markdown(html: str) -> str:
    """
    Convert HTML to Markdown, keeping as much structure as possible.

    Tries markdownify first (best at tables/links/lists), then html2text,
    then a very basic tag-stripping fallback so the script still works
    even with no conversion library installed.
    """
    # Preferred: markdownify
    try:
        from markdownify import markdownify as md

        return md(html, heading_style="ATX", strip=["script", "style"])
    except ImportError:
        pass

    # Fallback: html2text
    try:
        import html2text

        h = html2text.HTML2Text()
        h.body_width = 0          # don't hard-wrap lines
        h.ignore_links = False    # keep links
        h.ignore_images = False   # keep image references
        h.ignore_tables = False   # keep tables
        return h.handle(html)
    except ImportError:
        pass

    # Last resort: strip tags with the standard library. Loses formatting
    # but preserves the readable text content.
    print(
        "Warning: neither 'markdownify' nor 'html2text' is installed. "
        "Falling back to basic tag stripping (some structure will be lost). "
        "Run: pip install markdownify",
        file=sys.stderr,
    )
    from html.parser import HTMLParser

    class _Stripper(HTMLParser):
        def __init__(self):
            super().__init__()
            self.parts = []
            self._skip = False

        def handle_starttag(self, tag, attrs):
            if tag in ("script", "style"):
                self._skip = True
            if tag in ("p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4"):
                self.parts.append("\n")

        def handle_endtag(self, tag):
            if tag in ("script", "style"):
                self._skip = False

        def handle_data(self, data):
            if not self._skip:
                self.parts.append(data)

    stripper = _Stripper()
    stripper.feed(html)
    return "".join(stripper.parts)


def load_document(path: Path) -> str:
    """Read the file and return its content as Markdown/plain text."""
    if not path.exists():
        sys.exit(f"Error: input file not found: {path}")

    raw = path.read_text(encoding="utf-8", errors="replace")
    kind = detect_type(path)

    if kind == "html":
        print("Detected HTML — converting to Markdown...", file=sys.stderr)
        return html_to_markdown(raw)

    # Markdown and plain text are passed through unchanged.
    print(f"Detected {kind} — using content as-is.", file=sys.stderr)
    return raw


def load_schema(path: Path) -> dict:
    """Load and lightly validate the JSON schema file."""
    if not path.exists():
        sys.exit(f"Error: schema file not found: {path}")
    try:
        schema = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"Error: schema file is not valid JSON: {e}")
    if not isinstance(schema, dict):
        sys.exit("Error: schema must be a JSON object.")
    # Claude's tool input_schema expects an object at the top level.
    if schema.get("type") != "object":
        print(
            "Warning: top-level schema 'type' is not 'object'. Claude's "
            "structured-output tool works best with an object schema.",
            file=sys.stderr,
        )
    return schema


def extract(document: str, schema: dict, model: str) -> dict:
    """Call Claude and return the structured data as a dict."""
    try:
        import anthropic
    except ImportError:
        sys.exit("Error: the 'anthropic' package is not installed. Run: pip install anthropic")

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("Error: set the ANTHROPIC_API_KEY environment variable first.")

    client = anthropic.Anthropic()

    tool = {
        "name": "record_extracted_data",
        "description": (
            "Record the structured data extracted from the document. "
            "Every field must follow the provided schema. Use null or "
            "omit optional fields when the information is not present in "
            "the document — never invent values."
        ),
        "input_schema": schema,
    }

    system = (
        "You are a precise data-extraction engine. Read the document and "
        "extract the requested information, then call the "
        "'record_extracted_data' tool with the results. Only use facts "
        "stated in the document. Do not fabricate missing data."
    )

    message = client.messages.create(
        model=model,
        max_tokens=4096,
        system=system,
        tools=[tool],
        tool_choice={"type": "tool", "name": "record_extracted_data"},
        messages=[
            {
                "role": "user",
                "content": (
                    "Extract the structured data from the following document:\n\n"
                    "<document>\n" + document + "\n</document>"
                ),
            }
        ],
    )

    for block in message.content:
        if block.type == "tool_use" and block.name == "record_extracted_data":
            return block.input

    sys.exit("Error: Claude did not return structured data. Try a simpler schema or a clearer document.")


def main():
    parser = argparse.ArgumentParser(
        description="Extract structured data from an HTML, Markdown, or text document using Claude.",
    )
    parser.add_argument("--input", "-i", required=True, type=Path, help="Path to the input document.")
    parser.add_argument("--schema", "-s", required=True, type=Path, help="Path to the JSON schema file.")
    parser.add_argument("--output", "-o", type=Path, help="Where to write the JSON result. Prints to stdout if omitted.")
    parser.add_argument("--model", "-m", default=DEFAULT_MODEL, help=f"Claude model to use (default: {DEFAULT_MODEL}).")
    args = parser.parse_args()

    document = load_document(args.input)
    schema = load_schema(args.schema)
    result = extract(document, schema, args.model)

    output_json = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(output_json + "\n", encoding="utf-8")
        print(f"Wrote structured data to {args.output}", file=sys.stderr)
    else:
        print(output_json)


if __name__ == "__main__":
    main()
