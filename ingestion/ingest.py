def download_pdf():
    pdf_path = Path(settings.pdf_local_path)
    pdf_path.parent.mkdir(parents=True, exist_ok=True)

    if pdf_path.exists() and pdf_path.stat().st_size > 10000:
        print("PDF already downloaded")
        return str(pdf_path)

    print(f"Downloading PDF from {settings.pdf_url}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (X11; Linux x86_64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://konverge.ai/",
        "Connection": "keep-alive",
    }

    # try multiple times in case of network issues
    for attempt in range(3):
        try:
            print(f"Attempt {attempt + 1} of 3")
            response = requests.get(
                settings.pdf_url,
                timeout=120,
                headers=headers,
                stream=True,
                allow_redirects=True,
            )
            response.raise_for_status()

            total = 0
            with open(pdf_path, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=8192
                ):
                    if chunk:
                        f.write(chunk)
                        total += len(chunk)

            size = pdf_path.stat().st_size
            print(f"Downloaded {size} bytes")

            if size < 5000:
                print(f"File too small: {size} bytes")
                pdf_path.unlink()
                continue

            return str(pdf_path)

        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if pdf_path.exists():
                pdf_path.unlink()
            if attempt < 2:
                import time
                time.sleep(3)
            continue

    raise RuntimeError(
        "Failed to download PDF after 3 attempts. "
        "Check if the URL is accessible."
    )
