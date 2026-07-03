"""
Step 1 of Phase 1: pull every 8-K filed by your ticker basket in the lookback
window, and download the actual filing text -- including press-release
exhibits, not just the cover-page wrapper -- for each one.
"""

import os
import re
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings

import config

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

HEADERS = {"User-Agent": config.SEC_USER_AGENT}

EXHIBIT_NAME_PATTERN = re.compile(r"ex-?99", re.IGNORECASE)


def get_ticker_to_cik_map() -> dict:
    url = "https://www.sec.gov/files/company_tickers.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    raw = resp.json()
    return {row["ticker"]: str(row["cik_str"]).zfill(10) for row in raw.values()}


def get_recent_8ks(cik: str, ticker: str, cutoff_date: datetime) -> list[dict]:
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    if resp.status_code != 200:
        print(f"  [{ticker}] submissions lookup failed: {resp.status_code}")
        return []

    filings = resp.json().get("filings", {}).get("recent", {})
    forms = filings.get("form", [])
    dates = filings.get("filingDate", [])
    accessions = filings.get("accessionNumber", [])
    primary_docs = filings.get("primaryDocument", [])

    results = []
    for form, date_str, accession, primary_doc in zip(forms, dates, accessions, primary_docs):
        if form != "8-K":
            continue
        filing_date = datetime.strptime(date_str, "%Y-%m-%d")
        if filing_date < cutoff_date:
            continue
        results.append({
            "ticker": ticker, "cik": cik, "form_type": form,
            "filing_date": date_str, "accession_number": accession,
            "primary_document": primary_doc,
        })
    return results


def get_exhibit_documents(cik: str, accession_number: str) -> list[str]:
    cik_no_zeros = str(int(cik))
    accession_no_dashes = accession_number.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{accession_no_dashes}/index.json"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        if resp.status_code != 200:
            return []
        items = resp.json().get("directory", {}).get("item", [])
    except Exception:
        return []

    exhibit_files = []
    for item in items:
        name = item.get("name", "")
        item_type = item.get("type", "")
        if EXHIBIT_NAME_PATTERN.search(name) or EXHIBIT_NAME_PATTERN.search(item_type):
            exhibit_files.append(name)
    return exhibit_files


def download_document_text(cik: str, accession_number: str, document_name: str) -> str:
    cik_no_zeros = str(int(cik))
    accession_no_dashes = accession_number.replace("-", "")
    url = f"https://www.sec.gov/Archives/edgar/data/{cik_no_zeros}/{accession_no_dashes}/{document_name}"
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()

    if document_name.lower().endswith((".htm", ".html")):
        soup = BeautifulSoup(resp.text, "lxml")

        # Strip hidden iXBRL cover-page facts -- these aren't meant to be
        # visible, but get_text() ignores CSS and would flatten them into
        # the output as garbage otherwise.
        for hidden in soup.find_all(style=lambda v: v and "display:none" in v.replace(" ", "").lower()):
            hidden.decompose()
        for tag_name in ("ix:header", "ix:hidden"):
            for hidden_tag in soup.find_all(tag_name):
                hidden_tag.decompose()

        text = soup.get_text(separator=" ", strip=True)
    else:
        text = resp.text

    return text


def download_filing_text(cik: str, accession_number: str, primary_document: str) -> str:
    parts = [download_document_text(cik, accession_number, primary_document)]

    for exhibit_name in get_exhibit_documents(cik, accession_number):
        if exhibit_name == primary_document:
            continue
        try:
            exhibit_text = download_document_text(cik, accession_number, exhibit_name)
            parts.append(f"\n\n--- EXHIBIT: {exhibit_name} ---\n\n{exhibit_text}")
        except Exception as e:
            print(f"    (exhibit {exhibit_name} failed: {e})")
        time.sleep(config.SEC_REQUEST_DELAY_SECONDS)

    return "".join(parts)


def main():
    os.makedirs(config.RAW_FILINGS_DIR, exist_ok=True)
    cutoff_date = datetime.now() - timedelta(days=config.LOOKBACK_DAYS)

    print("Fetching ticker -> CIK map...")
    ticker_to_cik = get_ticker_to_cik_map()
    time.sleep(config.SEC_REQUEST_DELAY_SECONDS)

    all_filings = []
    for ticker in config.TICKERS:
        cik = ticker_to_cik.get(ticker)
        if cik is None:
            print(f"  [{ticker}] not found in SEC ticker map, skipping")
            continue

        print(f"[{ticker}] looking up 8-Ks since {cutoff_date.date()}...")
        filings = get_recent_8ks(cik, ticker, cutoff_date)
        print(f"  -> found {len(filings)} 8-K filings")
        all_filings.extend(filings)
        time.sleep(config.SEC_REQUEST_DELAY_SECONDS)

    print(f"\nTotal 8-Ks collected: {len(all_filings)}")
    print("Downloading filing text, including press-release exhibits (this is the slow part)...")

    for i, filing in enumerate(all_filings):
        text_path = f"{config.RAW_FILINGS_DIR}/{filing['ticker']}_{filing['accession_number']}.txt"
        if os.path.exists(text_path):
            filing["text_path"] = text_path
            filing["char_count"] = os.path.getsize(text_path)
            continue

        try:
            text = download_filing_text(filing["cik"], filing["accession_number"], filing["primary_document"])
            with open(text_path, "w", encoding="utf-8") as f:
                f.write(text)
            filing["text_path"] = text_path
            filing["char_count"] = len(text)
        except Exception as e:
            print(f"  failed on {filing['ticker']} {filing['accession_number']}: {e}")
            filing["text_path"] = None
            filing["char_count"] = 0

        time.sleep(config.SEC_REQUEST_DELAY_SECONDS)

        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(all_filings)} filings downloaded")

    df = pd.DataFrame(all_filings)
    df.to_parquet(config.FILINGS_METADATA_PATH, index=False)
    print(f"\nSaved metadata for {len(df)} filings to {config.FILINGS_METADATA_PATH}")
    print(df["ticker"].value_counts())


if __name__ == "__main__":
    main()