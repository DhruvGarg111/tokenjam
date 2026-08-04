"""Ingestion of the agent-config surface — everything that reaches an agent
before it does any work.

Three analyzers used to answer "what config does this user have" by walking the
filesystem themselves, each with its own enumeration, at analysis time:
``deadweight`` re-read ``~/.claude.json`` and every project's ``.mcp.json``,
``core/summarize/candidates`` re-globbed the catalog, and ``prompt_bloat``
re-globbed it a second time with a different helper. Nothing about any of those
files was ever STORED, so the same directory tree was walked three times per run
and no question about it could be answered without touching disk at all.

This module makes the walk a POPULATION step and the table the thing analyzers
read. The shape is deliberately the same as the rest of ``core``: a scan builds
:class:`ConfigRecord` values, a store persists them, and readers query the store.
The default store is in-memory, so a caller that wants today's behaviour and no
database (every direct unit-test call, the CLI's own scan) gets exactly that;
the registered analyzers pass a DuckDB-backed store so a run's config surface
outlives the run.

**What is stored.** Per record: what it is (``kind``), where it lives (``path``,
``root``, ``scope``), its identity within that root (``name`` — the SLOT, so the
same file under two worktrees is recognisably one file seen twice), its size in
bytes, its token count, a content hash, and when it was last seen. That is
enough to answer presence, size, drift and staleness without a stat call.

**Kinds.**

* ``instruction`` — the catalog's project files, project globs and global paths
  (``core/summarize/catalog``, itself backed by ``agent_files.toml``). This is
  the SINGLE source of truth for which paths count; nothing here re-lists them.
  Agent files, commands, skills and rules are instruction files whose
  ``subkind`` records which of those they are.
* ``hook`` — one record per hook command declared in a ``settings.json`` /
  ``settings.local.json``, keyed by event + matcher, because a hook is config
  that fires rather than config that is read.
* ``mcp_server`` — one record per (server name, declaring config file). Carries
  the server's own spec in ``detail`` and, once measured, the token size of the
  tool schemas it actually injects (``measured_tokens`` — see
  ``core/optimize/mcp_probe``). Measuring means starting the server, so the
  measurement is CACHED here against the spec hash and re-taken only when the
  spec changes.

Read-only, always: this module reads config files and never writes one.
"""
from __future__ import annotations

import glob as _glob
import hashlib
import json
import logging
import os
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from tokenjam.core.summarize.catalog import Catalog, load_catalog
from tokenjam.core.summarize.detect import CHARS_PER_TOKEN
from tokenjam.utils.time_parse import utcnow

if TYPE_CHECKING:  # pragma: no cover - typing only
    import duckdb

log = logging.getLogger(__name__)

# --- Vocabulary -------------------------------------------------------------

KIND_INSTRUCTION = "instruction"
KIND_HOOK = "hook"
KIND_MCP_SERVER = "mcp_server"

SCOPE_GLOBAL = "global"
SCOPE_PROJECT = "project"

#: Measurement states a record's ``measure_status`` can hold. ``""`` means the
#: question was never asked. Every other value is a positive statement about an
#: attempt, which is what keeps "never measured" distinguishable from "measured
#: and found small" — see ``core/optimize/mcp_probe``.
MEASURE_OK = "measured"
MEASURE_UNREACHABLE = "unreachable"
MEASURE_UNSUPPORTED = "unsupported"
MEASURE_SKIPPED = "skipped"

#: Settings files that can declare hooks and MCP servers at project scope. Same
#: list ``deadweight`` used inline; single-sourced here now that both the MCP and
#: the hook scan need it.
PROJECT_SETTINGS_RELPATHS = (
    ".claude/settings.json",
    ".claude/settings.local.json",
)
#: Plus the dedicated MCP file, which carries no hooks.
PROJECT_MCP_RELPATHS = (".mcp.json",) + PROJECT_SETTINGS_RELPATHS

#: Global settings file (hooks live here for a user-scoped install). Resolved
#: against an explicit home so a scoped run never reads the operator's real one.
GLOBAL_SETTINGS_RELPATH = ".claude/settings.json"
#: Global MCP config.
GLOBAL_MCP_RELPATH = ".claude.json"


def _subkind_for(path: str) -> str:
    """Which FLAVOUR of instruction file this is, from its slot.

    Derived from the path rather than declared, because the catalog states
    locations and the flavour is a property of the location. Used for
    presentation and for callers that want only agent definitions; never for
    deciding whether a file counts, which is the catalog's job alone.
    """
    posix = Path(path).as_posix()
    for marker, name in (
        ("/.claude/agents/", "agent"),
        ("/.claude/commands/", "command"),
        ("/.claude/skills/", "skill"),
        ("/.claude/rules/", "rule"),
    ):
        if marker in posix or posix.startswith(marker.lstrip("/")):
            return name
    return "instruction"


# --- The record -------------------------------------------------------------

@dataclass(frozen=True)
class ConfigRecord:
    """One ingested piece of agent config.

    ``config_id`` is derived, not assigned: the same slot rescanned produces the
    same id, so an upsert updates a row rather than accumulating history. Drift
    is visible through ``content_hash`` changing under a stable id, which is the
    question a consumer actually has ("did this file change since we priced
    it"), and it is why the hash is stored rather than only the size.
    """

    kind: str
    scope: str
    #: The project root this belongs to. ``""`` for global scope, which has none.
    root: str
    #: Identity WITHIN the root — the file's slot (``CLAUDE.md``,
    #: ``.claude/commands/ship.md``), a server name, or ``<event>:<matcher>``.
    name: str
    #: Absolute path of the file this was read from. For an MCP server or a hook
    #: that is the DECLARING config file, not a file of its own.
    path: str
    size_bytes: int = 0
    tokens: int = 0
    content_hash: str = ""
    last_seen: datetime = field(default_factory=utcnow)
    subkind: str = ""
    #: Free-form structured extras (an MCP server's spec, a hook's command).
    detail: dict[str, Any] = field(default_factory=dict)
    #: Independently MEASURED token size of what this record injects, when a
    #: measurement was taken. ``None`` means unmeasured, and no consumer may
    #: substitute a default for it — see ``measure_status``.
    measured_tokens: int | None = None
    measured_at: datetime | None = None
    measure_status: str = ""
    #: Scan-order position, so reading the table back reproduces the order the
    #: walk produced. Without it a store round-trip would silently reorder an
    #: analyzer's enumeration.
    seq: int = 0

    @property
    def config_id(self) -> str:
        raw = "\0".join((self.kind, self.scope, self.root, self.name, self.path))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:32]


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_obj(obj: Any) -> str:
    return _hash_text(json.dumps(obj, sort_keys=True, default=str))


def tokens_for_chars(chars: int) -> int:
    """Chars -> tokens on the repo's one shared constant.

    Deliberately the same ``CHARS_PER_TOKEN`` every other tj surface uses
    (``core/summarize/detect``) rather than a second ratio local to config
    files: two ratios would make a config file's token count incomparable with
    the transcript-measured counts it is reported beside.
    """
    return max(0, round(max(0, int(chars)) / CHARS_PER_TOKEN))


# --- Stores -----------------------------------------------------------------

@dataclass(frozen=True)
class MeasurementRow:
    """What a store remembers about one record's measurement.

    ``status == ""`` means the question was never asked, which is why this is a
    structure rather than a bare token count: "never measured", "measured and
    found small" and "tried and failed" are three different answers, and a
    consumer that can only see a number cannot tell them apart.
    """

    tokens: int | None = None
    status: str = ""
    at: datetime | None = None
    #: Extra measured quantities the caller wanted kept alongside the headline
    #: one (tool counts, the deferred-listing size). Persisted inside the
    #: record's ``detail`` JSON under :data:`MEASUREMENT_DETAIL_KEY`.
    extra: dict[str, Any] = field(default_factory=dict)


#: Where a measurement's extra quantities live inside ``detail``. Namespaced so
#: a measurement can never collide with a key the scan itself wrote (a server's
#: own ``command``, ``args``, ``env``).
MEASUREMENT_DETAIL_KEY = "measurement"


class AgentConfigStore:
    """Where ingested config lives. Two implementations, one contract."""

    def upsert(self, records: Sequence[ConfigRecord]) -> None:
        raise NotImplementedError

    def select(
        self,
        *,
        kind: str | None = None,
        scope: str | None = None,
        root: str | None = None,
        seen_at: datetime | None = None,
    ) -> list[ConfigRecord]:
        """Records matching every given filter, in scan order.

        ``seen_at`` restricts to a single population pass. A persistent store
        holds every root ever scanned, so an analyzer that read the whole table
        would price servers from a repo the current window never touched —
        filtering on the pass's own timestamp is what keeps a stored table
        answering the same question the live walk answered.
        """
        raise NotImplementedError

    def record_measurement(
        self, config_id: str, *, tokens: int | None, status: str, at: datetime,
        extra: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    def measurement_for(self, config_id: str) -> MeasurementRow:
        """What was last recorded for ``config_id``.

        An all-default :class:`MeasurementRow` when nothing was — never a zero,
        which would read as "measured and found empty".
        """
        raise NotImplementedError


class InMemoryAgentConfigStore(AgentConfigStore):
    """The default. Insertion-ordered, so a round-trip through it is
    order-preserving and a caller that passes no store sees exactly the
    enumeration the walk produced."""

    def __init__(self) -> None:
        self._rows: dict[str, ConfigRecord] = {}

    def upsert(self, records: Sequence[ConfigRecord]) -> None:
        for record in records:
            key = record.config_id
            prior = self._rows.get(key)
            # A rescan must not drop a measurement taken against an unchanged
            # spec: measuring means starting a server, and re-measuring on every
            # analysis pass is exactly what this store exists to avoid.
            if (
                prior is not None
                and record.measured_tokens is None
                and not record.measure_status
                and prior.content_hash == record.content_hash
            ):
                # The measurement's EXTRAS travel with it. A fresh scan builds
                # `detail` from the config file alone, so carrying the headline
                # token count forward while dropping the tool count and the
                # deferred size would leave a half-restored measurement that
                # reads as complete.
                detail = dict(record.detail)
                carried = prior.detail.get(MEASUREMENT_DETAIL_KEY)
                if isinstance(carried, dict):
                    detail[MEASUREMENT_DETAIL_KEY] = carried
                record = replace(
                    record,
                    measured_tokens=prior.measured_tokens,
                    measured_at=prior.measured_at,
                    measure_status=prior.measure_status,
                    detail=detail,
                )
            self._rows[key] = record

    def select(
        self,
        *,
        kind: str | None = None,
        scope: str | None = None,
        root: str | None = None,
        seen_at: datetime | None = None,
    ) -> list[ConfigRecord]:
        out = [
            r for r in self._rows.values()
            if (kind is None or r.kind == kind)
            and (scope is None or r.scope == scope)
            and (root is None or r.root == root)
            and (seen_at is None or r.last_seen == seen_at)
        ]
        out.sort(key=lambda r: (r.seq, r.path, r.name))
        return out

    def record_measurement(
        self, config_id: str, *, tokens: int | None, status: str, at: datetime,
        extra: dict[str, Any] | None = None,
    ) -> None:
        prior = self._rows.get(config_id)
        if prior is None:
            return
        detail = dict(prior.detail)
        detail[MEASUREMENT_DETAIL_KEY] = dict(extra or {})
        self._rows[config_id] = replace(
            prior, measured_tokens=tokens, measure_status=status, measured_at=at,
            detail=detail,
        )

    def measurement_for(self, config_id: str) -> MeasurementRow:
        prior = self._rows.get(config_id)
        if prior is None:
            return MeasurementRow()
        extra = prior.detail.get(MEASUREMENT_DETAIL_KEY)
        return MeasurementRow(
            tokens=prior.measured_tokens, status=prior.measure_status,
            at=prior.measured_at, extra=extra if isinstance(extra, dict) else {},
        )


class DuckDBAgentConfigStore(AgentConfigStore):
    """The persistent one, over ``agent_config_files`` (migration 22).

    Parameterised SQL only (Critical Rule 7) and ``TIMESTAMPTZ``/``JSON`` column
    types (Critical Rule 1). Never opens its own connection — the caller owns it.
    """

    def __init__(self, conn: "duckdb.DuckDBPyConnection") -> None:
        self.conn = conn

    _COLUMNS = (
        "config_id, kind, scope, root, name, path, size_bytes, tokens, "
        "content_hash, last_seen, subkind, detail, measured_tokens, "
        "measured_at, measure_status, seq"
    )

    def upsert(self, records: Sequence[ConfigRecord]) -> None:
        for record in records:
            prior = self.measurement_for(record.config_id)
            prior_hash = self._content_hash(record.config_id)
            measured_tokens = record.measured_tokens
            measured_at = record.measured_at
            measure_status = record.measure_status
            detail = dict(record.detail)
            if (
                measured_tokens is None
                and not measure_status
                and prior_hash == record.content_hash
            ):
                measured_tokens, measure_status, measured_at = (
                    prior.tokens, prior.status, prior.at,
                )
                if prior.extra:
                    detail[MEASUREMENT_DETAIL_KEY] = prior.extra
            self.conn.execute(
                "DELETE FROM agent_config_files WHERE config_id = $1",
                [record.config_id],
            )
            self.conn.execute(
                f"INSERT INTO agent_config_files ({self._COLUMNS}) VALUES "
                "($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16)",
                [
                    record.config_id, record.kind, record.scope, record.root,
                    record.name, record.path, record.size_bytes, record.tokens,
                    record.content_hash, record.last_seen, record.subkind,
                    json.dumps(detail, sort_keys=True, default=str),
                    measured_tokens, measured_at, measure_status, record.seq,
                ],
            )

    def _content_hash(self, config_id: str) -> str | None:
        row = self.conn.execute(
            "SELECT content_hash FROM agent_config_files WHERE config_id = $1",
            [config_id],
        ).fetchone()
        return row[0] if row else None

    def select(
        self,
        *,
        kind: str | None = None,
        scope: str | None = None,
        root: str | None = None,
        seen_at: datetime | None = None,
    ) -> list[ConfigRecord]:
        where: list[str] = []
        params: list[Any] = []
        for column, value in (
            ("kind", kind), ("scope", scope), ("root", root), ("last_seen", seen_at),
        ):
            if value is not None:
                params.append(value)
                where.append(f"{column} = ${len(params)}")
        sql = f"SELECT {self._COLUMNS} FROM agent_config_files"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY seq, path, name"
        return [_row_to_record(row) for row in self.conn.execute(sql, params).fetchall()]

    def record_measurement(
        self, config_id: str, *, tokens: int | None, status: str, at: datetime,
        extra: dict[str, Any] | None = None,
    ) -> None:
        row = self.conn.execute(
            "SELECT detail FROM agent_config_files WHERE config_id = $1", [config_id],
        ).fetchone()
        if not row:
            return
        try:
            detail = json.loads(row[0]) if row[0] else {}
        except (TypeError, ValueError):
            detail = {}
        if not isinstance(detail, dict):
            detail = {}
        detail[MEASUREMENT_DETAIL_KEY] = dict(extra or {})
        self.conn.execute(
            "UPDATE agent_config_files SET measured_tokens = $1, measure_status = $2, "
            "measured_at = $3, detail = $4 WHERE config_id = $5",
            [tokens, status, at, json.dumps(detail, sort_keys=True, default=str), config_id],
        )

    def measurement_for(self, config_id: str) -> MeasurementRow:
        row = self.conn.execute(
            "SELECT measured_tokens, measure_status, measured_at, detail "
            "FROM agent_config_files WHERE config_id = $1",
            [config_id],
        ).fetchone()
        if not row:
            return MeasurementRow()
        try:
            detail = json.loads(row[3]) if row[3] else {}
        except (TypeError, ValueError):
            detail = {}
        extra = detail.get(MEASUREMENT_DETAIL_KEY) if isinstance(detail, dict) else None
        return MeasurementRow(
            tokens=int(row[0]) if row[0] is not None else None,
            status=str(row[1] or ""),
            at=row[2],
            extra=extra if isinstance(extra, dict) else {},
        )


def _row_to_record(row: Sequence[Any]) -> ConfigRecord:
    try:
        detail = json.loads(row[11]) if row[11] else {}
    except (TypeError, ValueError):
        detail = {}
    return ConfigRecord(
        kind=str(row[1]), scope=str(row[2]), root=str(row[3] or ""),
        name=str(row[4] or ""), path=str(row[5] or ""),
        size_bytes=int(row[6] or 0), tokens=int(row[7] or 0),
        content_hash=str(row[8] or ""), last_seen=row[9],
        subkind=str(row[10] or ""), detail=detail if isinstance(detail, dict) else {},
        measured_tokens=int(row[12]) if row[12] is not None else None,
        measured_at=row[13], measure_status=str(row[14] or ""),
        seq=int(row[15] or 0),
    )


def store_for(conn: "duckdb.DuckDBPyConnection | None") -> AgentConfigStore:
    """A DuckDB-backed store when there is a connection, else an in-memory one.

    The degrade is deliberate and silent-by-design in only one direction: a
    caller with no database still gets a working store and today's behaviour,
    never an error. It never fabricates the reverse — nothing here invents a
    connection.
    """
    if conn is None:
        return InMemoryAgentConfigStore()
    return DuckDBAgentConfigStore(conn)


# --- Population: the filesystem walk ----------------------------------------

def _expand_home(raw: str, home: Path | None) -> str:
    """Expand a leading ``~`` against ``home``, or the real home when None.

    Same rule ``core/summarize/candidates`` applies, and for the same reason: a
    scoped run redirects ``~`` itself, because the catalog's globals span several
    agent homes.
    """
    if home is None:
        return os.path.expanduser(raw)
    if raw == "~":
        return str(home)
    if raw.startswith("~/"):
        return str(home / raw[2:])
    return raw


def _stat_file(path: Path) -> tuple[int, str, str] | None:
    """``(size_bytes, content_hash, text)`` for a readable file, else None."""
    try:
        if not path.is_file():
            return None
        data = path.read_bytes()
    except OSError:
        return None
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = ""
    return len(data), hashlib.sha256(data).hexdigest(), text


def _slot_for(path: Path, root: Path | None) -> str:
    if root is None:
        return path.name
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def catalog_global_paths(
    home: Path | None = None, catalog: "Catalog | None" = None,
) -> list[Path]:
    """Every catalog global path that exists, globs expanded, in catalog order.

    The one enumeration of the catalog's globals. ``prompt_bloat`` and
    ``candidates`` each had their own before this; they now both come through
    here, which is what makes "the catalog is the source of truth" true rather
    than merely intended.

    ``catalog`` is passed rather than looked up so a caller that isolates the
    catalog keeps isolating it. Each consumer holds its own module-level
    ``load_catalog`` reference and its tests patch THAT; resolving the catalog
    here instead would silently route every scan back to the real
    ``~/.claude`` while those tests still read as isolated — a harness that
    cannot see what it is measuring, which returns a plausible number rather
    than an error.
    """
    cat = catalog if catalog is not None else load_catalog()
    out: list[Path] = []
    for raw in cat.global_paths:
        expanded = _expand_home(raw, home)
        if any(ch in expanded for ch in "*?["):
            out.extend(Path(x) for x in sorted(_glob.glob(expanded, recursive=True)))
        else:
            out.append(Path(expanded))
    return out


def catalog_project_paths(
    root: Path, *, extra_exts: Iterable[str] = (), catalog: "Catalog | None" = None,
) -> list[Path]:
    """Catalog names + globs at ``root``, plus root-level files with an
    extension in ``extra_exts`` when the caller has widened the net.

    ``catalog`` is the caller's, for the isolation reason in
    :func:`catalog_global_paths`."""
    cat = catalog if catalog is not None else load_catalog()
    exts = {e for e in extra_exts if e}
    out: list[Path] = []
    for name in sorted(cat.project_files):
        out.append(root / name)
    for pattern in cat.project_globs:
        out.extend(sorted(root.glob(pattern)))
    if exts:
        try:
            out.extend(
                p for p in sorted(root.iterdir())
                if p.is_file() and p.suffix.lower() in exts
            )
        except OSError:
            pass
    return out


def _instruction_records(
    paths: Sequence[Path],
    *,
    scope: str,
    root: Path | None,
    seen_at: datetime,
    start_seq: int,
) -> list[ConfigRecord]:
    out: list[ConfigRecord] = []
    seen_paths: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen_paths:
            continue
        seen_paths.add(key)
        stat = _stat_file(path)
        if stat is None:
            continue
        size, digest, text = stat
        out.append(ConfigRecord(
            kind=KIND_INSTRUCTION,
            scope=scope,
            root=str(root) if root is not None else "",
            name=_slot_for(path, root),
            path=key,
            size_bytes=size,
            tokens=tokens_for_chars(len(text)),
            content_hash=digest,
            last_seen=seen_at,
            subkind=_subkind_for(key),
            seq=start_seq + len(out),
        ))
    return out


def scan_instruction_files(
    *,
    roots: Sequence[Path | str] = (),
    home: Path | None = None,
    include_global: bool = True,
    extra_exts: Iterable[str] = (),
    seen_at: datetime | None = None,
    extra_paths: Sequence[tuple[Path, str, Path | None]] = (),
    catalog: "Catalog | None" = None,
) -> list[ConfigRecord]:
    """Every catalog-known instruction file under ``roots`` (and the globals).

    ``extra_paths`` is ``(path, scope, root)`` for files a caller enumerated some
    other way — a ``--recursive`` walk, an explicitly named file. They are
    ingested on the same terms as everything else rather than bypassing the
    store, so no consumer has to know which door a file came in through.

    ``catalog`` is the caller's own, for the isolation reason in
    :func:`catalog_global_paths`.
    """
    at = seen_at or utcnow()
    records: list[ConfigRecord] = []
    if include_global:
        records.extend(_instruction_records(
            catalog_global_paths(home, catalog), scope=SCOPE_GLOBAL, root=None,
            seen_at=at, start_seq=len(records),
        ))
    for raw_root in roots:
        project_root = Path(raw_root).expanduser()
        if not project_root.is_dir():
            continue
        records.extend(_instruction_records(
            catalog_project_paths(project_root, extra_exts=extra_exts, catalog=catalog),
            scope=SCOPE_PROJECT, root=project_root, seen_at=at,
            start_seq=len(records),
        ))
    for path, scope, extra_root in extra_paths:
        records.extend(_instruction_records(
            [path], scope=scope, root=extra_root, seen_at=at,
            start_seq=len(records),
        ))
    return records


def _read_json_safe(path: Path) -> dict:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _settings_paths(
    roots: Sequence[Path | str], *, claude_home: Path | None, relpaths: Sequence[str],
    global_relpath: str,
) -> list[tuple[Path, str, str]]:
    """``(config file, scope, root)`` for every settings file to read, globals
    first then project roots in sorted order — the order ``deadweight`` already
    depended on, kept explicit so the store round-trip reproduces it.

    ``root`` is returned as the caller's OWN string, not a re-rendered ``Path``.
    A recorded session cwd is matched against this value downstream, and
    normalising it here would silently stop a cwd carrying a trailing separator
    from matching the server it really does reach.
    """
    out: list[tuple[Path, str, str]] = []
    home = claude_home if claude_home is not None else Path.home()
    global_path = home / global_relpath
    if global_path.is_file():
        out.append((global_path, SCOPE_GLOBAL, ""))
    for raw in sorted(str(r) for r in roots):
        if not raw:
            continue
        base = Path(raw)
        if not base.is_dir():
            continue
        for rel in relpaths:
            path = base / rel
            if path.is_file():
                out.append((path, SCOPE_PROJECT, raw))
    return out


def scan_mcp_servers(
    *,
    roots: Sequence[Path | str] = (),
    claude_home: Path | None = None,
    seen_at: datetime | None = None,
) -> list[ConfigRecord]:
    """One record per (server name, declaring config file).

    Read-only. ``detail`` carries the server's own spec plus a ``spec_hash`` — the
    key a cached schema measurement is valid against, so changing a server's
    command or args invalidates its measurement while a rescan of an unchanged
    one reuses it (starting a server is not free).
    """
    at = seen_at or utcnow()
    records: list[ConfigRecord] = []
    for path, scope, root in _settings_paths(
        roots, claude_home=claude_home,
        relpaths=PROJECT_MCP_RELPATHS, global_relpath=GLOBAL_MCP_RELPATH,
    ):
        data = _read_json_safe(path)
        servers = data.get("mcpServers")
        if not isinstance(servers, dict):
            continue
        for name in sorted(servers):
            if not str(name).strip():
                continue
            spec = servers[name] if isinstance(servers[name], dict) else {}
            detail = dict(spec)
            detail["spec_hash"] = _hash_obj(spec)
            serialized = json.dumps(spec, sort_keys=True, default=str)
            records.append(ConfigRecord(
                kind=KIND_MCP_SERVER,
                scope=scope,
                root=root,
                name=str(name),
                path=str(path),
                size_bytes=len(serialized),
                tokens=tokens_for_chars(len(serialized)),
                content_hash=detail["spec_hash"],
                last_seen=at,
                subkind=str(spec.get("type") or ("stdio" if spec.get("command") else "")),
                detail=detail,
                seq=len(records),
            ))
    return records


def scan_hooks(
    *,
    roots: Sequence[Path | str] = (),
    claude_home: Path | None = None,
    seen_at: datetime | None = None,
) -> list[ConfigRecord]:
    """One record per hook command declared in a settings file.

    A hook is config that FIRES rather than config that is read, so it is stored
    keyed by ``<event>:<matcher>:<ordinal>`` and its command text is what gets
    sized. The ordinal is what keeps two hooks on the same event and matcher from
    collapsing onto one row.
    """
    at = seen_at or utcnow()
    records: list[ConfigRecord] = []
    for path, scope, root in _settings_paths(
        roots, claude_home=claude_home,
        relpaths=PROJECT_SETTINGS_RELPATHS, global_relpath=GLOBAL_SETTINGS_RELPATH,
    ):
        hooks = _read_json_safe(path).get("hooks")
        if not isinstance(hooks, dict):
            continue
        for event in sorted(hooks):
            groups = hooks[event]
            if not isinstance(groups, list):
                continue
            for group in groups:
                if not isinstance(group, dict):
                    continue
                matcher = str(group.get("matcher") or "*")
                entries = group.get("hooks")
                if not isinstance(entries, list):
                    continue
                for ordinal, entry in enumerate(entries):
                    if not isinstance(entry, dict):
                        continue
                    command = str(entry.get("command") or "")
                    detail = {
                        "event": event, "matcher": matcher,
                        "type": str(entry.get("type") or ""),
                        "command": command,
                        "timeout": entry.get("timeout"),
                    }
                    records.append(ConfigRecord(
                        kind=KIND_HOOK,
                        scope=scope,
                        root=root,
                        name=f"{event}:{matcher}:{ordinal}",
                        path=str(path),
                        size_bytes=len(command),
                        tokens=tokens_for_chars(len(command)),
                        content_hash=_hash_obj(detail),
                        last_seen=at,
                        subkind=str(event),
                        detail=detail,
                        seq=len(records),
                    ))
    return records


def ingest_agent_config(
    store: AgentConfigStore,
    *,
    roots: Sequence[Path | str] = (),
    home: Path | None = None,
    claude_home: Path | None = None,
    kinds: Sequence[str] = (KIND_INSTRUCTION, KIND_HOOK, KIND_MCP_SERVER),
    include_global: bool = True,
    extra_exts: Iterable[str] = (),
    extra_paths: Sequence[tuple[Path, str, Path | None]] = (),
    seen_at: datetime | None = None,
    catalog: "Catalog | None" = None,
) -> datetime:
    """Walk, then store. Returns the pass's timestamp.

    That return value is the handle a reader uses: every record this pass wrote
    carries it as ``last_seen``, so ``store.select(seen_at=...)`` is exactly the
    population the walk found and never a stale root from a previous window.
    """
    at = seen_at or utcnow()
    records: list[ConfigRecord] = []
    if KIND_INSTRUCTION in kinds:
        records.extend(scan_instruction_files(
            roots=roots, home=home, include_global=include_global,
            extra_exts=extra_exts, extra_paths=extra_paths, seen_at=at,
            catalog=catalog,
        ))
    if KIND_MCP_SERVER in kinds:
        records.extend(scan_mcp_servers(
            roots=roots, claude_home=claude_home, seen_at=at,
        ))
    if KIND_HOOK in kinds:
        records.extend(scan_hooks(
            roots=roots, claude_home=claude_home, seen_at=at,
        ))
    store.upsert(records)
    return at


__all__ = [
    "GLOBAL_MCP_RELPATH",
    "GLOBAL_SETTINGS_RELPATH",
    "KIND_HOOK",
    "KIND_INSTRUCTION",
    "KIND_MCP_SERVER",
    "MEASUREMENT_DETAIL_KEY",
    "MEASURE_OK",
    "MEASURE_SKIPPED",
    "MEASURE_UNREACHABLE",
    "MEASURE_UNSUPPORTED",
    "PROJECT_MCP_RELPATHS",
    "PROJECT_SETTINGS_RELPATHS",
    "SCOPE_GLOBAL",
    "SCOPE_PROJECT",
    "AgentConfigStore",
    "ConfigRecord",
    "MeasurementRow",
    "DuckDBAgentConfigStore",
    "InMemoryAgentConfigStore",
    "catalog_global_paths",
    "catalog_project_paths",
    "ingest_agent_config",
    "scan_hooks",
    "scan_instruction_files",
    "scan_mcp_servers",
    "store_for",
    "tokens_for_chars",
]
