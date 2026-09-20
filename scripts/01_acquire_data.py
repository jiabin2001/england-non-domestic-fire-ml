from fireml.acquire import acquire_and_cache


if __name__ == "__main__":
    frame, metadata = acquire_and_cache()
    print(f"Cached {len(frame):,} rows from sheet {metadata['data_sheet']!r}.")
    print(f"SHA-256: {metadata['sha256']}")

