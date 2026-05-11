"""Сбор постов Reddit через PRAW.

PRAW требует регистрации приложения на https://www.reddit.com/prefs/apps
и кладёт client_id / client_secret в .env. Reddit API ограничивает listing
до 1000 постов - для глубокой истории нужен Pushshift или Kaggle-дамп,
а нам тут хватит свежей ~1000 для иллюстрации.
"""

from __future__ import annotations

import argparse
import os
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

from src.utils.config import PROJECT_ROOT, get_data_config
from src.utils.logging_setup import get_logger

logger = get_logger(__name__)

# подгружаем .env, оттуда возьмём REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET
load_dotenv()


def fetch_subreddit_posts(
    subreddit_name: str,
    reddit,
    limit: int = 1000,
) -> pd.DataFrame:
    """Забирает посты из одного сабреддита (объединение new + hot, без дублей)."""
    subreddit = reddit.subreddit(subreddit_name)
    rows: list[dict] = []

    # объединяем потоки new() и hot() - у них есть пересечения, но в сумме покрытие лучше
    seen_ids: set[str] = set()
    for listing in (subreddit.new(limit=limit), subreddit.hot(limit=limit)):
        for post in listing:
            if post.id in seen_ids:
                continue
            seen_ids.add(post.id)
            rows.append(
                {
                    "id": post.id,
                    "subreddit": subreddit_name,
                    "title": post.title,
                    "selftext": post.selftext,
                    "score": post.score,
                    "upvote_ratio": post.upvote_ratio,
                    "num_comments": post.num_comments,
                    "created_utc": datetime.utcfromtimestamp(post.created_utc),
                    "author": str(post.author) if post.author else None,
                    "url": post.url,
                    "permalink": f"https://reddit.com{post.permalink}",
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Скачать посты Reddit для тем нефти и BTC")
    parser.add_argument("--limit", type=int, default=1000, help="Максимум постов на сабреддит")
    args = parser.parse_args()

    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "hse-yp/0.1")

    if not (client_id and client_secret):
        # без ключей API дёргать нельзя - просто завершаемся
        logger.error("REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET не заданы в .env - пропуск.")
        return

    # praw - опциональная зависимость, импортируем только когда реально нужно
    try:
        import praw
    except ModuleNotFoundError:
        logger.error("praw не установлен. Запустите: pip install praw")
        return

    reddit = praw.Reddit(
        client_id=client_id,
        client_secret=client_secret,
        user_agent=user_agent,
    )
    # read-only - нам никаких писательных прав не нужно
    reddit.read_only = True

    cfg = get_data_config()
    out_dir = PROJECT_ROOT / cfg["texts_storage"]["raw_dir"] / "reddit"
    out_dir.mkdir(parents=True, exist_ok=True)

    for topic_key, subs in cfg["texts"]["reddit"]["subreddits"].items():
        all_dfs = []
        for sub_name in subs:
            logger.info("[%s] скачиваю r/%s", topic_key, sub_name)
            try:
                df = fetch_subreddit_posts(sub_name, reddit, limit=args.limit)
                all_dfs.append(df)
            except Exception as exc:
                # один проблемный сабреддит не должен валить весь сбор
                logger.error("r/%s упал: %s", sub_name, exc)
        if all_dfs:
            combined = pd.concat(all_dfs, ignore_index=True).drop_duplicates(subset=["id"])
            path = out_dir / f"reddit_{topic_key}_{datetime.utcnow().date()}.parquet"
            combined.to_parquet(path)
            logger.info("[%s] сохранено %d постов -> %s", topic_key, len(combined), path.relative_to(PROJECT_ROOT))


if __name__ == "__main__":
    main()
