import logging
from collections.abc import Generator
from typing import Union

import httpx

from src.common.client import HttpClient
from src.indexers.polymarket.models import Market

logger = logging.getLogger(__name__)

GAMMA_API_URL = "https://gamma-api.polymarket.com"


class PolymarketClient:
    def __init__(self, gamma_url: str = GAMMA_API_URL):
        self.gamma_url = gamma_url
        self.http = HttpClient(rate_limit=10)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.http.close()

    def close(self):
        self.http.close()

    def get_markets(self, limit: int = 500, offset: int = 0, **kwargs) -> list[Market]:
        params = {"limit": limit, "offset": offset, **kwargs}
        data: Union[dict, list] = self.http.get(f"{self.gamma_url}/markets", params=params)
        if isinstance(data, list):
            return [Market.from_dict(m) for m in data]
        return [Market.from_dict(m) for m in data.get("markets", data)]

    def iter_markets(
        self, limit: int = 500, offset: int = 0, **kwargs
    ) -> Generator[tuple[list[Market], int], None, None]:
        # The gamma API caps each page at ~100 regardless of the requested
        # `limit`, so stopping when len(markets) < limit (the requested size)
        # ends the backfill after just the first page. Only an empty page
        # means there's nothing left.
        #
        # Offset-based pagination is also capped by the API itself (~2100
        # deep) — beyond that it returns 422 "offset too large, use
        # /markets/keyset for deeper pagination". We stop cleanly there
        # rather than crash; getting the most recent ~2000+ markets matching
        # the filter is far better than none. Pass closed=True via kwargs to
        # backfill resolved markets (the only ones the win-rate query uses)
        # instead of the newest/still-open markets the unfiltered default
        # would return.
        current_offset = offset

        while True:
            try:
                markets = self.get_markets(limit=limit, offset=current_offset, **kwargs)
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code == 422:
                    logger.info(
                        "Stopping pagination at offset=%d: %s", current_offset, exc.response.text,
                    )
                    yield [], -1
                    break
                raise

            if not markets:
                yield [], -1
                break

            next_offset = current_offset + len(markets)
            yield markets, next_offset

            current_offset = next_offset
