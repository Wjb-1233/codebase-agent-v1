from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class AnalysisRecord(Base):
    __tablename__ = "analysis_record"

    id: Mapped[int] = mapped_column(primary_key=True)
    repo_url: Mapped[str] = mapped_column(nullable=False)
    count: Mapped[int] = mapped_column(nullable=False)
    create_at: Mapped[str] = mapped_column(nullable=False)
