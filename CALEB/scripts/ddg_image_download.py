#!/usr/bin/env python3
import argparse
import asyncio
import hashlib
import os
import re
import sys
import time
import random
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import aiohttp
from PIL import Image
import yaml
from tqdm import tqdm
from slugify import slugify
import imagehash
from io import BytesIO
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

try:
    from duckduckgo_search import DDGS
    from duckduckgo_search.exceptions import RatelimitException
except Exception as e:
    print("Failed to import duckduckgo_search. Make sure requirements are installed.", file=sys.stderr)
    raise

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sha1_hex(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8", errors="ignore")).hexdigest()


def pick_extension_from_url_or_content_type(url: str, content_type: Optional[str]) -> Optional[str]:
    url_path = url.split("?")[0].split("#")[0]
    ext = Path(url_path).suffix.lower()
    if ext in VALID_EXTENSIONS:
        return ext
    if content_type:
        if "jpeg" in content_type:
            return ".jpg"
        if "png" in content_type:
            return ".png"
    return None


async def validate_image_bytes(img_bytes: bytes, min_size: int) -> Optional[Tuple[Image.Image, str]]:
    try:
        img = Image.open(BytesIO(img_bytes))
        img.verify()  # quick check
        img = Image.open(BytesIO(img_bytes)).convert("RGB")
        width, height = img.size
        if min(width, height) < min_size:
            return None
        return img, "ok"
    except Exception:
        return None


@retry(reraise=True, stop=stop_after_attempt(3), wait=wait_exponential(multiplier=0.7, min=0.7, max=6), retry=retry_if_exception_type(aiohttp.ClientError))
async def fetch_bytes(session: aiohttp.ClientSession, url: str, timeout: int) -> Tuple[Optional[bytes], Optional[str]]:
    async with session.get(url, timeout=timeout) as resp:
        if resp.status != 200:
            return None, resp.headers.get("Content-Type")
        content_type = resp.headers.get("Content-Type")
        return await resp.read(), content_type


async def download_one(
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    url: str,
    out_dir: Path,
    min_size: int,
    seen_hashes: Set[str],
    pbar: tqdm,
    timeout: int,
) -> Optional[Path]:
    async with semaphore:
        try:
            data, content_type = await fetch_bytes(session, url, timeout)
            if not data:
                pbar.update(1)
                return None
            valid = await validate_image_bytes(data, min_size)
            if not valid:
                pbar.update(1)
                return None
            img, _ = valid
            percept_hash = str(imagehash.phash(img))
            if percept_hash in seen_hashes:
                pbar.update(1)
                return None
            seen_hashes.add(percept_hash)

            ext = pick_extension_from_url_or_content_type(url, content_type)
            if ext is None:
                pbar.update(1)
                return None
            fname = f"{sha1_hex(url)}{ext}"
            out_path = out_dir / fname
            if out_path.exists():
                pbar.update(1)
                return None
            img.save(out_path, quality=90)
            pbar.update(1)
            return out_path
        except Exception:
            pbar.update(1)
            return None


def load_queries(yaml_path: Path) -> Dict[str, List[str]]:
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # YAML keys: category -> list of keywords
    return {str(cat): [str(q) for q in lst] for cat, lst in data.items()}


def ddg_image_urls(keyword: str, max_results: int) -> List[str]:
    urls: List[str] = []
    attempts = 5
    for attempt in range(1, attempts + 1):
        try:
            with DDGS() as ddgs:
                for r in ddgs.images(
                    keywords=keyword,
                    safesearch="moderate",
                    size=None,
                    type_image=None,
                    layout=None,
                    license_image=None,
                    max_results=max_results,
                ):
                    u = r.get("image") or r.get("thumbnail") or r.get("url")
                    if not u:
                        continue
                    urls.append(u)
            break
        except RatelimitException:
            sleep_s = min(10, 2 * attempt) + random.uniform(0.2, 0.8)
            time.sleep(sleep_s)
            continue
        except Exception:
            # Brief backoff on any other transient error
            time.sleep(1.0 + 0.3 * attempt)
            continue
    # de-dupe while preserving order
    seen: Set[str] = set()
    uniq = []
    for u in urls:
        if u not in seen:
            uniq.append(u)
            seen.add(u)
    return uniq


async def run_download(
    queries_file: Path,
    base_dir: Path,
    limit_per_keyword: int,
    concurrent_downloads: int,
    min_size: int,
    timeout: int,
) -> None:
    queries = load_queries(queries_file)
    ensure_dir(base_dir)

    connector = aiohttp.TCPConnector(limit_per_host=max(4, concurrent_downloads // 2))
    headers = {"User-Agent": USER_AGENT, "Accept": "image/*,*/*;q=0.8"}

    async with aiohttp.ClientSession(headers=headers, connector=connector) as session:
        for category, keywords in queries.items():
            category_dir = base_dir / category
            ensure_dir(category_dir)
            seen_hashes: Set[str] = set()
            print(f"\n----- Category: {category} -----")
            for keyword in keywords:
                print(f"Fetching URLs for: {keyword}")
                # Keep max_results modest to avoid rate limits
                urls = ddg_image_urls(keyword, max_results=max(10, int(limit_per_keyword * 1.5)))
                if not urls:
                    print("No URLs found (rate limited or none available).")
                    # Small pause between keywords to be polite
                    await asyncio.sleep(1.0)
                    continue
                urls = urls[: max(limit_per_keyword * 2, limit_per_keyword + 5)]
                sem = asyncio.Semaphore(concurrent_downloads)
                with tqdm(total=len(urls), desc=slugify(keyword)[:40]) as pbar:
                    tasks = [
                        download_one(
                            session=session,
                            semaphore=sem,
                            url=u,
                            out_dir=category_dir,
                            min_size=min_size,
                            seen_hashes=seen_hashes,
                            pbar=pbar,
                            timeout=timeout,
                        )
                        for u in urls
                    ]
                    _ = await asyncio.gather(*tasks)
                # polite pause between keywords
                await asyncio.sleep(0.5)
        print("\nImage downloads complete.")


def main():
    parser = argparse.ArgumentParser(description="Download images by category/keyword using DuckDuckGo.")
    parser.add_argument("--queries_file", type=Path, default=Path("data/queries.yaml"))
    parser.add_argument("--base_dir", type=Path, default=Path("dataset/train"))
    parser.add_argument("--limit", type=int, default=35, help="Target results per keyword (before filtering)")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--min_size", type=int, default=256, help="Minimum shorter-side pixels")
    parser.add_argument("--timeout", type=int, default=20)

    args = parser.parse_args()
    os.makedirs(args.base_dir, exist_ok=True)
    try:
        asyncio.run(
            run_download(
                queries_file=args.queries_file,
                base_dir=args.base_dir,
                limit_per_keyword=args.limit,
                concurrent_downloads=args.concurrency,
                min_size=args.min_size,
                timeout=args.timeout,
            )
        )
    except KeyboardInterrupt:
        print("Interrupted by user.")


if __name__ == "__main__":
    main()