import os
from datetime import datetime

from sqlalchemy import create_engine, select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

from codebase_agent.backend.models import Base, AnalysisRecord

# 按连接 URL 复用 engine，避免每次操作都新建连接
_engines: dict[str, Engine] = {}


def build_database_url(db_path: str | None = None) -> str:
    """构造数据库连接 URL。

    优先级：环境变量 DATABASE_URL > db_path 参数 > 报错。
    本地开发传 db_path，Docker/生产设置 DATABASE_URL。
    """
    url = os.getenv("DATABASE_URL")
    if url:
        return url
    if db_path is None:
        raise ValueError("必须提供 db_path 或设置 DATABASE_URL 环境变量")
    return f"sqlite:///{db_path}"


def get_engine(db_path: str | None = None) -> Engine:
    """返回 SQLAlchemy Engine（按连接 URL 复用）。

    SQLite 默认禁止跨线程使用连接；FastAPI 在线程池执行端点，
    所以对 sqlite:// URL 显式设置 check_same_thread=False。
    PostgreSQL 等 URL 不接受该参数，需按 URL 类型区分。
    """
    url = build_database_url(db_path)
    engine = _engines.get(url)
    if engine is None:
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        engine = create_engine(url, connect_args=connect_args)
        _engines[url] = engine
    return engine


def init_db(db_path: str) -> None:
    engine = get_engine(db_path)
    Base.metadata.create_all(engine)


def save_analysis(db_path: str, repo_url: str, count: int) -> int:
    engine = get_engine(db_path)
    with Session(engine) as session:
        try:
            analysis = AnalysisRecord(
                repo_url=repo_url,
                count=count,
                create_at=datetime.now(),
            )
            session.add(analysis)
            session.commit()
            return analysis.id
        except Exception:
            session.rollback()
            raise


def list_recent_analyses(db_path: str, limit: int = 10) -> list[dict]:
    engine = get_engine(db_path)
    with Session(engine) as session:
        stmt = (
            select(AnalysisRecord)
            .order_by(AnalysisRecord.create_at.desc(), AnalysisRecord.id.desc())
            .limit(limit)
        )
        analyses = session.execute(stmt).scalars().all()
        return [
            {
                "id": r.id,
                "repo_url": r.repo_url,
                "count": r.count,
                "create_at": r.create_at,
            }
            for r in analyses
        ]
