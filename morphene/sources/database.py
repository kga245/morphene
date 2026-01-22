"""
Morphene Source Tracking Database

SQLite-based database for tracking audio source metadata, including:
- Origin (recorded, licensed, commissioned)
- License type and restrictions
- Total duration collected
- Quality rating (1-5)
- Processing applied
- File locations
"""

import os
import json
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional, List, Dict, Any


class SourceOrigin(Enum):
    """Origin of an audio source."""
    RECORDED = "recorded"       # Field recording or original recording
    LICENSED = "licensed"       # Licensed from a sample library
    COMMISSIONED = "commissioned"  # Commissioned performance
    SYNTHESIZED = "synthesized"   # Synthesized or sampled from hardware


class SourceType(Enum):
    """Type of audio source."""
    MUSICAL = "musical"
    NONMUSICAL = "nonmusical"


@dataclass
class AudioSource:
    """
    Represents an audio source for a Morphene instrument.

    Attributes:
        id: Unique identifier
        instrument_number: Instrument this source belongs to
        source_type: Musical or non-musical
        name: Human-readable name
        origin: How the source was acquired
        license_type: License type (e.g., "CC0", "royalty-free", "exclusive")
        license_restrictions: Any restrictions on use
        total_duration_sec: Total duration of collected audio in seconds
        quality_rating: Quality rating from 1-5
        processing_applied: List of processing steps applied
        file_paths: List of file paths for this source
        notes: Additional notes
        created_at: When this record was created
        updated_at: When this record was last updated
    """
    instrument_number: int
    source_type: SourceType
    name: str
    origin: SourceOrigin
    id: Optional[int] = None
    license_type: str = ""
    license_restrictions: str = ""
    total_duration_sec: float = 0.0
    quality_rating: int = 0
    processing_applied: List[str] = field(default_factory=list)
    file_paths: List[str] = field(default_factory=list)
    notes: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        d = asdict(self)
        d['source_type'] = self.source_type.value
        d['origin'] = self.origin.value
        d['processing_applied'] = json.dumps(self.processing_applied)
        d['file_paths'] = json.dumps(self.file_paths)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'AudioSource':
        """Create from dictionary."""
        d = d.copy()
        d['source_type'] = SourceType(d['source_type'])
        d['origin'] = SourceOrigin(d['origin'])
        d['processing_applied'] = json.loads(d.get('processing_applied', '[]'))
        d['file_paths'] = json.loads(d.get('file_paths', '[]'))
        return cls(**d)

    @property
    def duration_hours(self) -> float:
        """Return duration in hours."""
        return self.total_duration_sec / 3600

    @property
    def meets_minimum_duration(self) -> bool:
        """Check if source meets minimum 1.5 hour requirement."""
        return self.duration_hours >= 1.5


class SourceDatabase:
    """
    SQLite database for tracking Morphene audio sources.
    """

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS sources (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        instrument_number INTEGER NOT NULL,
        source_type TEXT NOT NULL,
        name TEXT NOT NULL,
        origin TEXT NOT NULL,
        license_type TEXT DEFAULT '',
        license_restrictions TEXT DEFAULT '',
        total_duration_sec REAL DEFAULT 0.0,
        quality_rating INTEGER DEFAULT 0,
        processing_applied TEXT DEFAULT '[]',
        file_paths TEXT DEFAULT '[]',
        notes TEXT DEFAULT '',
        created_at TEXT,
        updated_at TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_instrument ON sources(instrument_number);
    CREATE INDEX IF NOT EXISTS idx_source_type ON sources(source_type);

    CREATE TABLE IF NOT EXISTS reference_clips (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        source_id INTEGER NOT NULL,
        file_path TEXT NOT NULL,
        duration_sec REAL NOT NULL,
        description TEXT DEFAULT '',
        created_at TEXT,
        FOREIGN KEY (source_id) REFERENCES sources(id)
    );
    """

    def __init__(self, db_path: str):
        """
        Initialize the database.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path) or '.', exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(self.SCHEMA)

    def _get_connection(self) -> sqlite3.Connection:
        """Get a database connection with row factory."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def add_source(self, source: AudioSource) -> int:
        """
        Add a new audio source.

        Args:
            source: AudioSource to add

        Returns:
            ID of the new source
        """
        now = datetime.now().isoformat()
        source.created_at = now
        source.updated_at = now

        data = source.to_dict()
        del data['id']  # Let SQLite auto-generate

        columns = ', '.join(data.keys())
        placeholders = ', '.join(['?' for _ in data])

        with self._get_connection() as conn:
            cursor = conn.execute(
                f"INSERT INTO sources ({columns}) VALUES ({placeholders})",
                list(data.values())
            )
            return cursor.lastrowid

    def update_source(self, source: AudioSource) -> None:
        """
        Update an existing audio source.

        Args:
            source: AudioSource with id set
        """
        if source.id is None:
            raise ValueError("Source must have an id to update")

        source.updated_at = datetime.now().isoformat()
        data = source.to_dict()
        del data['id']
        del data['created_at']

        set_clause = ', '.join([f"{k} = ?" for k in data.keys()])

        with self._get_connection() as conn:
            conn.execute(
                f"UPDATE sources SET {set_clause} WHERE id = ?",
                list(data.values()) + [source.id]
            )

    def get_source(self, source_id: int) -> Optional[AudioSource]:
        """
        Get a source by ID.

        Args:
            source_id: Source ID

        Returns:
            AudioSource or None
        """
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM sources WHERE id = ?",
                (source_id,)
            ).fetchone()

            if row:
                return AudioSource.from_dict(dict(row))
        return None

    def get_sources_for_instrument(self, instrument_number: int) -> List[AudioSource]:
        """
        Get all sources for an instrument.

        Args:
            instrument_number: Instrument number

        Returns:
            List of AudioSource objects
        """
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM sources WHERE instrument_number = ?",
                (instrument_number,)
            ).fetchall()

            return [AudioSource.from_dict(dict(row)) for row in rows]

    def get_all_sources(self) -> List[AudioSource]:
        """Get all sources in the database."""
        with self._get_connection() as conn:
            rows = conn.execute("SELECT * FROM sources").fetchall()
            return [AudioSource.from_dict(dict(row)) for row in rows]

    def delete_source(self, source_id: int) -> None:
        """Delete a source by ID."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM sources WHERE id = ?", (source_id,))

    def add_reference_clip(
        self,
        source_id: int,
        file_path: str,
        duration_sec: float,
        description: str = ""
    ) -> int:
        """
        Add a reference clip for auditing.

        Args:
            source_id: ID of the parent source
            file_path: Path to the reference clip
            duration_sec: Duration in seconds
            description: Optional description

        Returns:
            ID of the new reference clip
        """
        now = datetime.now().isoformat()

        with self._get_connection() as conn:
            cursor = conn.execute(
                """INSERT INTO reference_clips
                   (source_id, file_path, duration_sec, description, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (source_id, file_path, duration_sec, description, now)
            )
            return cursor.lastrowid

    def get_reference_clips(self, source_id: int) -> List[Dict[str, Any]]:
        """Get reference clips for a source."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM reference_clips WHERE source_id = ?",
                (source_id,)
            ).fetchall()
            return [dict(row) for row in rows]

    def get_instrument_status(self, instrument_number: int) -> Dict[str, Any]:
        """
        Get collection status for an instrument.

        Args:
            instrument_number: Instrument number

        Returns:
            Dict with status information
        """
        sources = self.get_sources_for_instrument(instrument_number)

        musical_sources = [s for s in sources if s.source_type == SourceType.MUSICAL]
        nonmusical_sources = [s for s in sources if s.source_type == SourceType.NONMUSICAL]

        musical_duration = sum(s.total_duration_sec for s in musical_sources)
        nonmusical_duration = sum(s.total_duration_sec for s in nonmusical_sources)

        return {
            'instrument_number': instrument_number,
            'musical_duration_hours': musical_duration / 3600,
            'nonmusical_duration_hours': nonmusical_duration / 3600,
            'musical_ready': musical_duration / 3600 >= 1.5,
            'nonmusical_ready': nonmusical_duration / 3600 >= 1.5,
            'ready_for_training': (musical_duration / 3600 >= 1.5 and
                                   nonmusical_duration / 3600 >= 1.5),
            'musical_sources_count': len(musical_sources),
            'nonmusical_sources_count': len(nonmusical_sources),
        }

    def get_collection_summary(self) -> Dict[str, Any]:
        """Get summary of all source collection progress."""
        sources = self.get_all_sources()

        # Group by instrument
        by_instrument = {}
        for source in sources:
            if source.instrument_number not in by_instrument:
                by_instrument[source.instrument_number] = []
            by_instrument[source.instrument_number].append(source)

        ready_count = 0
        partial_count = 0
        not_started_count = 0

        for inst_num in range(1, 51):
            status = self.get_instrument_status(inst_num)
            if status['ready_for_training']:
                ready_count += 1
            elif status['musical_sources_count'] > 0 or status['nonmusical_sources_count'] > 0:
                partial_count += 1
            else:
                not_started_count += 1

        return {
            'total_instruments': 50,
            'ready_for_training': ready_count,
            'in_progress': partial_count,
            'not_started': not_started_count,
            'total_sources': len(sources),
            'total_duration_hours': sum(s.total_duration_sec for s in sources) / 3600,
        }


# Global database instance
_database: Optional[SourceDatabase] = None


def get_database(db_path: Optional[str] = None) -> SourceDatabase:
    """
    Get or create the global database instance.

    Args:
        db_path: Path to database file. If None, uses default.

    Returns:
        SourceDatabase instance
    """
    global _database

    if db_path is None:
        # Default path in project directory
        db_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
            'data',
            'sources.db'
        )

    if _database is None or _database.db_path != db_path:
        _database = SourceDatabase(db_path)

    return _database
