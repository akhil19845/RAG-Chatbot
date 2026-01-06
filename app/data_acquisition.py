# app/data_acquisition.py
import logging
import time
import re
from pathlib import Path
from typing import List, Optional
import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.colors import black
from .config import BASE_DIR, DATA_FILE

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


TARGET_WORD_COUNT = 28000
MACHINE_LEARNING_URL = "https://en.wikipedia.org/wiki/Machine_learning"
BASE_WIKI_URL = "https://en.wikipedia.org"


def fetch_html(url: str) -> Optional[str]:
    logger.debug("Fetching URL: %s", url)
    time.sleep(0.5)
    try:
        headers = {'User-Agent': 'RAG-Pipeline-Project/1.0 (akhil.gorthi@gmail.com)'}
        resp = requests.get(url, headers=headers, timeout=10)
        resp.raise_for_status()
        return resp.text
    except requests.RequestException as e:
        logger.warning("Error fetching %s: %s", url, e)
        return None


def extract_main_content(html_content: str, url: str, depth: int) -> str:

    if not html_content:
        return ""

    soup = BeautifulSoup(html_content, "html.parser")
    content_div = soup.find("div", {"id": "mw-content-text"})
    if not content_div:
        return ""

    content_text = []
    skip_sections = ["References", "See also", "External links", "Footnotes", "Notes", "Further reading"]

    for element in content_div.find_all(['p', 'h2', 'h3', 'ul', 'ol', 'table']):

        if element.name in ('h2', 'h3'):
            header = element.get_text(strip=True).replace('[edit]', '').strip()
            if not header:
                continue
            if any(s in header for s in skip_sections):
                break
            if element.name == 'h2':
                content_text.append(f"\n\n## {header}\n")
            else:
                content_text.append(f"\n\n### {header}\n")

        elif element.name == 'p':
            text = element.get_text(strip=False)
            # remove bracketed citations and simple numeric locators
            text = re.sub(r'\[.*?\]|:\s*\d+', ' ', text)
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                content_text.append(text)

    tables_html = str(content_div)
    table_text = []
    try:
        tables = pd.read_html(StringIO(tables_html), flavor='bs4', attrs={'class': 'wikitable'})
        for i, df in enumerate(tables):
            desc = f"\n\n--- TABLE {i+1} START ---\nSource URL: {url}\n"
            desc += "Columns: " + ", ".join(df.columns.astype(str)) + ".\n"
            for _, row in df.head(5).iterrows():
                row_str = " | ".join(f"{col}: {val if pd.notna(val) else 'N/A'}" for col, val in row.items())
                desc += f"Row: {row_str}\n"
            desc += "--- TABLE END ---\n"
            table_text.append(desc)
    except ValueError:
        pass
    except Exception as e:
        logger.debug("Table parsing error (ignored): %s", e)

    return "\n\n".join(content_text + table_text)


def scrape_wikipedia_topic_tree(start_url: str = MACHINE_LEARNING_URL,max_depth: int = 2,word_count_limit: int = TARGET_WORD_COUNT) -> str:

    full_corpus = {}
    visited_urls = {start_url}
    queue = [(start_url, 0)]
    current_word_count = 0

    while queue and current_word_count < word_count_limit:
        current_url, depth = queue.pop(0)
        html = fetch_html(current_url)
        if not html:
            continue

        page_text = extract_main_content(html, current_url, depth)
        if not page_text:
            continue

        if current_url not in full_corpus:
            content_to_add = page_text
            if depth == 2:
                # only keep intro paragraphs for depth 2
                content_to_add = "\n\n".join(page_text.split("\n\n")[:2])
            full_corpus[current_url] = content_to_add
            current_word_count += len(content_to_add.split())
            logger.info("Accumulated words: %d/%d", current_word_count, word_count_limit)

        if depth < max_depth:
            soup = BeautifulSoup(html, "html.parser")
            content_div = soup.find("div", {"id": "mw-content-text"})
            if not content_div:
                continue
            for link in content_div.find_all("a", href=True):
                href = link["href"]
                if href.startswith("/wiki/") and ':' not in href and '#' not in href:
                    title_part = href.split('/')[-1]
                    if title_part and title_part[0].isupper():
                        title_lower = title_part.lower()
                        if any(tok in title_lower for tok in ("learning", "network", "algorithm", "data")):
                            next_url = BASE_WIKI_URL + href
                            if next_url not in visited_urls:
                                visited_urls.add(next_url)
                                queue.append((next_url, depth + 1))
                                if current_word_count >= 0.75 * word_count_limit:
                                    logger.info("Approaching word count cap; halting deep link discovery from current page.")
                                    break

    final_text = "\n\n".join(full_corpus.values())
    final_text = re.sub(r'\n\s*\n', '\n\n', final_text).strip()
    logger.info("Data acquisition complete. Estimated words: %d", len(final_text.split()))
    return final_text


def convert_text_to_pdf_reportlab(input_txt_path: Path, output_pdf_path: Path, title: str = "", overwrite: bool = False) -> Path:

    input_txt_path = Path(input_txt_path)
    output_pdf_path = Path(output_pdf_path)

    if output_pdf_path.exists() and not overwrite:
        logger.info("PDF already exists at %s (skip).", output_pdf_path)
        return output_pdf_path

    if not input_txt_path.exists():
        raise FileNotFoundError(f"Input text file not found: {input_txt_path}")

    with input_txt_path.open("r", encoding="utf8") as fh:
        full_text = fh.read()

    doc = SimpleDocTemplate(str(output_pdf_path), pagesize=letter,
                            leftMargin=72, rightMargin=72,
                            topMargin=72, bottomMargin=72)

    styles = getSampleStyleSheet()
    Story = []

    styles.add(ParagraphStyle(name='ContentBody',
                              parent=styles['Normal'],
                              fontSize=10,
                              leading=11))
    styles.add(ParagraphStyle(name='ContentH2',
                              parent=styles.get('Heading2', styles['Normal']),
                              fontName='Helvetica-Bold',
                              fontSize=16,
                              spaceBefore=18,
                              spaceAfter=6,
                              textColor=black))
    styles.add(ParagraphStyle(name='ContentH3',
                              parent=styles.get('Heading3', styles['Normal']),
                              fontName='Helvetica-Bold',
                              fontSize=12,
                              spaceBefore=10,
                              spaceAfter=4,
                              textColor=black))

    Story.append(Paragraph("Introduction", styles.get('Title', styles['Normal'])))
    Story.append(Spacer(1, 12))

    blocks = full_text.split('\n\n')

    for block in blocks:
        if not block.strip():
            continue

        if block.startswith("## "):
            heading_text = block.replace("## ", "").upper()
            Story.append(Paragraph(heading_text, styles['ContentH2']))
        elif block.startswith("### "):
            heading_text = block.replace("### ", "").upper()
            Story.append(Paragraph(heading_text, styles['ContentH3']))
        elif block.startswith("--- TABLE"):
            Story.append(Paragraph(block, styles['ContentBody']))
        else:
            Story.append(Paragraph(block, styles['ContentBody']))

        Story.append(Spacer(1, 4))

    try:
        doc.build(Story)
        logger.info("Saved PDF to %s", output_pdf_path)
    except Exception as e:
        logger.exception("Error building PDF: %s", e)
        raise

    return output_pdf_path


def acquire_and_prepare_data(sources: Optional[List[str]] = None,txt_out: Optional[Path] = None,pdf_out: Optional[Path] = None,overwrite: bool = False,max_depth: int = 2,word_count_limit: int = TARGET_WORD_COUNT,) -> Path:

    txt_out = Path(txt_out or (BASE_DIR / "Data" / "Scrapped_Data.txt"))
    pdf_out = Path(pdf_out or DATA_FILE)

    if txt_out.exists() and not overwrite:
        logger.info("Text corpus already exists at %s (skipping scrape).", txt_out)
    else:
        if sources:
            pieces = []
            for url in sources:
                html = fetch_html(url)
                if not html:
                    continue
                pieces.append(extract_main_content(html, url, depth=0))
            final_text = "\n\n".join(pieces)
        else:
            final_text = scrape_wikipedia_topic_tree(start_url=MACHINE_LEARNING_URL, max_depth=max_depth, word_count_limit=word_count_limit)

        txt_out.parent.mkdir(parents=True, exist_ok=True)
        with txt_out.open("w", encoding="utf8") as fh:
            fh.write(final_text)
        logger.info("Saved scraped text to %s", txt_out)

    pdf_path = convert_text_to_pdf_reportlab(txt_out, pdf_out, title="Machine Learning Wikipedia Corpus", overwrite=overwrite)
    return pdf_path


if __name__ == "__main__":
    try:
        pdf = acquire_and_prepare_data(overwrite=False)
        logger.info("Pipeline finished. PDF created at: %s", pdf)
    except Exception as exc:
        logger.exception("Data acquisition pipeline failed: %s", exc)