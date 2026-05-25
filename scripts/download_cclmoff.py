from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime, timezone
from http.cookiejar import CookieJar
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import HTTPCookieProcessor, ProxyHandler, Request, build_opener


ARTICLE_PAGE = "https://figshare.com/articles/dataset/CCLMoff_A_CRISPR_Cas9_System_Off-target_Prediction_Tool_Using_Language_Model/27080566"
DOWNLOAD_URL = "https://figshare.com/ndownloader/articles/27080566/versions/2"
DEFAULT_OUT = Path("data/cclmoff/09212024_CCLMoff_dataset.csv")
DEFAULT_LOG = Path("results/audits/cclmoff_download_log.txt")
DEFAULT_PROXY = "http://127.0.0.1:7897"
MIN_VALID_SIZE = 600 * 1024 * 1024

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
    "Accept-Encoding": "identity",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def append_log(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for line in lines:
            handle.write(line.rstrip() + "\n")


def configure_proxy(use_default_proxy: bool) -> dict[str, str]:
    proxy_override = os.environ.get("CCLMOFF_PROXY_URL")
    if proxy_override:
        os.environ["HTTP_PROXY"] = proxy_override
        os.environ["HTTPS_PROXY"] = proxy_override
        os.environ["http_proxy"] = proxy_override
        os.environ["https_proxy"] = proxy_override
    if use_default_proxy:
        os.environ.setdefault("HTTP_PROXY", DEFAULT_PROXY)
        os.environ.setdefault("HTTPS_PROXY", DEFAULT_PROXY)
        os.environ.setdefault("http_proxy", DEFAULT_PROXY)
        os.environ.setdefault("https_proxy", DEFAULT_PROXY)
    proxy_http = os.environ.get("HTTP_PROXY") or os.environ.get("http_proxy")
    proxy_https = os.environ.get("HTTPS_PROXY") or os.environ.get("https_proxy")
    proxies = {}
    if proxy_http:
        proxies["http"] = proxy_http
    if proxy_https:
        proxies["https"] = proxy_https
    return proxies


def build_http_opener(proxies: dict[str, str]):
    cookie_jar = CookieJar()
    opener = build_opener(ProxyHandler(proxies), HTTPCookieProcessor(cookie_jar))
    return opener, cookie_jar


def open_with_headers(opener, url: str, timeout: int):
    request = Request(url, headers=HEADERS)
    return opener.open(request, timeout=timeout)


def looks_like_html(path: Path) -> bool:
    with path.open("rb") as handle:
        head = handle.read(200).lower()
    return b"<html" in head or b"<!doctype" in head


def validate_download(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, "file missing"
    size = path.stat().st_size
    if size < MIN_VALID_SIZE:
        html = looks_like_html(path)
        return False, f"file too small: {size} bytes; html={html}"
    if looks_like_html(path):
        return False, "file appears to be HTML"
    return True, f"valid size: {size} bytes"


def download(args: argparse.Namespace) -> int:
    out_path: Path = args.output
    log_path: Path = args.log
    out_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    proxies = configure_proxy(args.use_default_proxy)
    append_log(
        log_path,
        [
            "",
            f"=== CCLMoff download attempt {utc_now()} ===",
            f"article_page={ARTICLE_PAGE}",
            f"download_url={DOWNLOAD_URL}",
            f"output={out_path}",
            f"proxies={proxies or 'none'}",
            f"headers.User-Agent={HEADERS['User-Agent']}",
        ],
    )

    if out_path.exists():
        ok, reason = validate_download(out_path)
        append_log(log_path, [f"existing_file={out_path}", f"existing_validation={ok}: {reason}"])
        if ok and not args.force:
            print(f"Existing valid file: {out_path}")
            return 0

    opener, cookie_jar = build_http_opener(proxies)
    try:
        page_response = open_with_headers(opener, ARTICLE_PAGE, args.page_timeout)
        page_response.read(1024)
        append_log(
            log_path,
            [
                f"page_status={page_response.status}",
                f"page_final_url={page_response.url}",
                f"page_content_type={page_response.headers.get('content-type')}",
                f"page_cookies={[cookie.name for cookie in cookie_jar]}",
            ],
        )

        response = open_with_headers(opener, DOWNLOAD_URL, args.download_timeout)
        append_log(
            log_path,
            [
                f"download_status={response.status}",
                f"download_final_url={response.url}",
                f"download_content_type={response.headers.get('content-type')}",
                f"download_content_length={response.headers.get('content-length')}",
            ],
        )
        if response.status != 200:
            preview = response.read(500)
            append_log(log_path, [f"download_failed_preview={preview!r}"])
            print(f"Download failed with status {response.status}; see {log_path}")
            return 2

        tmp_path = out_path.with_suffix(out_path.suffix + ".download.tmp")
        failed_path = out_path.with_suffix(out_path.suffix + ".download_failed")
        downloaded = 0
        with tmp_path.open("wb") as handle:
            while True:
                chunk = response.read(args.chunk_size)
                if not chunk:
                    break
                if chunk:
                    handle.write(chunk)
                    downloaded += len(chunk)
        append_log(log_path, [f"downloaded_bytes={downloaded}", f"tmp_path={tmp_path}"])
        ok, reason = validate_download(tmp_path)
        append_log(log_path, [f"validation={ok}: {reason}"])
        if not ok:
            if failed_path.exists():
                failed_path.unlink()
            shutil.move(str(tmp_path), str(failed_path))
            print(f"Downloaded invalid file moved to {failed_path}; see {log_path}")
            return 3
        tmp_path.replace(out_path)
        print(f"Downloaded valid CCLMoff CSV: {out_path}")
        return 0
    except HTTPError as exc:
        preview = exc.read(500)
        append_log(log_path, [f"http_error={exc.code}: {exc.reason}", f"http_error_preview={preview!r}"])
        print(f"HTTP error: {exc.code} {exc.reason}; see {log_path}")
        return 4
    except URLError as exc:
        append_log(log_path, [f"url_error={exc.reason!r}"])
        print(f"URL error: {exc.reason!r}; see {log_path}")
        return 4
    except Exception as exc:  # noqa: BLE001 - this is an audit script.
        append_log(log_path, [f"exception={exc.__class__.__name__}: {exc}"])
        print(f"Download exception: {exc.__class__.__name__}: {exc}; see {log_path}")
        return 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the public CCLMoff Figshare dataset with proxy and browser UA.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--use-default-proxy", action="store_true", help="Set proxy env vars to http://127.0.0.1:7897 if not already set.")
    parser.add_argument("--proxy-url", default="", help="Override HTTP proxy URL, e.g. http://127.0.0.1:7897.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--page-timeout", type=int, default=60)
    parser.add_argument("--download-timeout", type=int, default=180)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    if parsed_args.proxy_url:
        os.environ["CCLMOFF_PROXY_URL"] = parsed_args.proxy_url
    raise SystemExit(download(parsed_args))
