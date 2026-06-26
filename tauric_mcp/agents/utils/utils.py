from dataclasses import dataclass
from datetime import datetime, timedelta

@dataclass
class NewsItem:
    """新闻项目数据结构"""
    title: str
    content: str
    source: str
    publish_time: datetime
    url: str
    urgency: str  # high, medium, low
    relevance_score: float