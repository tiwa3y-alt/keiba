"""SQLAlchemy ORM models for keiba database."""

from sqlalchemy import (
    Column,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Text,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Race(Base):
    __tablename__ = "races"

    race_id = Column(Text, primary_key=True)  # "202405050811"
    race_name = Column(Text)
    race_date = Column(Date, index=True)
    venue = Column(Text)  # 東京/中山/阪神/京都/...
    course_type = Column(Text)  # 芝/ダート/障害
    distance = Column(Integer)
    track_condition = Column(Text)  # 良/稍重/重/不良
    weather = Column(Text)
    grade = Column(Text, index=True)  # G1/G2/G3
    race_class = Column(Text)
    head_count = Column(Integer)
    prize_1st = Column(Integer)  # 万円

    results = relationship("RaceResult", back_populates="race")
    payouts = relationship("Payout", back_populates="race")
    odds_history = relationship("OddsHistory", back_populates="race")


class RaceResult(Base):
    __tablename__ = "race_results"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(Text, ForeignKey("races.race_id"), index=True)
    horse_id = Column(Text, index=True)
    finish_order = Column(Integer)  # 着順 (0 = 取消/除外)
    frame_number = Column(Integer)  # 枠番
    horse_number = Column(Integer)  # 馬番
    horse_name = Column(Text)
    sex_age = Column(Text)  # 牡3
    weight_carried = Column(Float)  # 斤量
    jockey_id = Column(Text, index=True)
    jockey_name = Column(Text)
    trainer_id = Column(Text, index=True)
    trainer_name = Column(Text)
    finish_time = Column(Float)  # 秒
    margin = Column(Text)  # 着差
    passing_order = Column(Text)  # "3-3-2-1"
    last_3f = Column(Float)  # 上がり3F (秒)
    horse_weight = Column(Integer)
    weight_change = Column(Integer)
    odds = Column(Float)  # 単勝オッズ
    popularity = Column(Integer)

    race = relationship("Race", back_populates="results")


class Horse(Base):
    __tablename__ = "horses"

    horse_id = Column(Text, primary_key=True)
    horse_name = Column(Text)
    birth_date = Column(Date)
    sex = Column(Text)  # 牡/牝/セン
    sire_id = Column(Text)
    sire_name = Column(Text)
    dam_id = Column(Text)
    dam_name = Column(Text)
    broodmare_sire_id = Column(Text)
    broodmare_sire_name = Column(Text)
    owner = Column(Text)
    breeder = Column(Text)


class Payout(Base):
    __tablename__ = "payouts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(Text, ForeignKey("races.race_id"), index=True)
    bet_type = Column(Text)  # 単勝/複勝/馬連/ワイド/馬単/三連複/三連単
    combination = Column(Text)  # "3" or "3-5" or "3-5-8"
    payout = Column(Integer)  # 円
    popularity = Column(Integer)

    race = relationship("Race", back_populates="payouts")


class OddsHistory(Base):
    __tablename__ = "odds_history"

    id = Column(Integer, primary_key=True, autoincrement=True)
    race_id = Column(Text, ForeignKey("races.race_id"), index=True)
    horse_number = Column(Integer)
    timestamp = Column(DateTime)
    win_odds = Column(Float)
    place_odds_min = Column(Float)
    place_odds_max = Column(Float)

    race = relationship("Race", back_populates="odds_history")
