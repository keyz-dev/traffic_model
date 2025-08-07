#!/usr/bin/env python3
import argparse
import asyncio
import os
import sys
from pathlib import Path
from typing import List, Optional, Set

import aiohttp
from PIL import Image
from io import BytesIO
from slugify import slugify
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential

try:
    from playwright.async_api import async_playwright
except Exception as e:
    print("Playwright is not installed. Install requirements and run 'playwright install'.", file=sys.stderr)
    raise

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/127.0.0.0 Safari/537.36"
)
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}


async def collect_image_urls(keyword: str, max_urls: int, headless: bool = True, login: bool = False) -> List[str]:
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=headless)
        context = await browser.new_context(user_agent=USER_AGENT, viewport={"width": 1280, "height": 2000})
        page = await context.new_page()

        if login:
            email = os.environ.get("PINTEREST_EMAIL")
            password = os.environ.get("PINTEREST_PASSWORD")
            if not email or not password:
                print("PINTEREST_EMAIL/PINTEREST_PASSWORD not set; continuing without login.")
            else:
                await page.goto("https://www.pinterest.com/login/")
                await page.wait_for_timeout(1500)
                await page.fill('input[name="id"]', email)
                await page.fill('input[name="password"]', password)
                await page.click('button[type="submit"]')
                await page.wait_for_timeout(3000)

        search_url = "https://www.pinterest.com/search/pins/?q=" + keyword.replace(" ", "%20")
        await page.goto(search_url)
        await page.wait_for_timeout(2000)

        collected: List[str] = []
        seen: Set[str] = set()

        async def scrape_once():
            imgs = await page.query_selector_all('img[src*="i.pinimg.com"]')
            for img in imgs:
                src = await img.get_attribute("src")
                if src and src not in seen:
                    seen.add(src)
                    collected.append(src)
            return len(imgs)

        # Scroll and scrape loop
        while len(collected) < max_urls:
            await scrape_once()
            await page.evaluate("window.scrollBy(0, document.body.scrollHeight)")
            await page.wait_for_timeout(1200)
            # Attempt to click "Show more" if present
            try:
                button = await page.query_selector('button:has-text("Show more")')
                if button:
                    await button.click()
                    await page.wait_for_timeout(1000)
            except Exception:
                pass
            if len(collected) >= max_urls:
                break
        await browser.close()
        return collected[:max_urls]


async def download_images(urls: List[str], out_dir: Path, min_size: int, timeout: int) -> int:
    out_dir.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": USER_AGENT, "Accept": "image/*,*/*;q=0.8"}

    async with aiohttp.ClientSession(headers=headers) as session:
        success = 0
        with tqdm(total=len(urls), desc="download") as pbar:
            for url in urls:
                try:
                    async with session.get(url, timeout=timeout) as resp:
                        if resp.status != 200:
                            pbar.update(1)
                            continue
                        data = await resp.read()
                        img = Image.open(BytesIO(data))
                        img.verify()
                        img = Image.open(BytesIO(data)).convert("RGB")
                        w, h = img.size
                        if min(w, h) < min_size:
                            pbar.update(1)
                            continue
                        ext = ".jpg"
                        fname = slugify(url)[:60] + ext
                        (out_dir / fname).write_bytes(data)
                        success += 1
                except Exception:
                    pass
                finally:
                    pbar.update(1)
        return success


async def main_async(args):
    urls = await collect_image_urls(args.keyword, args.limit, headless=not args.show, login=args.login)
    saved = await download_images(urls, args.output_dir, args.min_size, args.timeout)
    print(f"Saved {saved} images to {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Pinterest image scraper (headless).")
    parser.add_argument("keyword", type=str, help="Search term")
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--output_dir", type=Path, default=Path("dataset/pinterest"))
    parser.add_argument("--min_size", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=20)
    parser.add_argument("--login", action="store_true", help="Attempt login with PINTEREST_EMAIL/PASSWORD")
    parser.add_argument("--show", action="store_true", help="Run with visible browser window")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("Interrupted by user.")


if __name__ == "__main__":
    main()