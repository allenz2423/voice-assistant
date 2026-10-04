import os
import re
import shutil
from pathlib import Path
from typing import Optional, Dict, List, Any, Tuple

class SkillManager:
    """Manages modular capability skills (Markdown context files) for desktop environments,
    window managers, and system automation tools."""

    def __init__(self, custom_skills_dir: Optional[Path] = None):
        self.builtin_skills_dir = Path(__file__).resolve().parent.parent.parent / "skills"
        self.user_skills_dir = custom_skills_dir or (Path.home() / ".config" / "adam" / "skills")
        self._index_dirty: bool = True
        self._indexed_skills: Dict[str, Dict[str, Any]] = {}
        self._bm25: Optional[Any] = None
        self._indexed_file_signature: tuple[tuple[str, int, int], ...] = ()

    def get_skill_paths(self) -> List[Path]:
        """Returns all search directories for skills in order of priority (user custom first)."""
        paths = []
        if self.user_skills_dir.exists():
            paths.append(self.user_skills_dir)
        if self.builtin_skills_dir.exists():
            paths.append(self.builtin_skills_dir)
        return paths

    def _skill_file_signature(self) -> tuple[tuple[str, int, int], ...]:
        """Tracks Markdown file additions, removals, and edits between matching queries."""
        entries = []
        for directory in self.get_skill_paths():
            for path in directory.glob("*.md"):
                try:
                    if path.is_file():
                        stat = path.stat()
                        entries.append((str(path.resolve()), stat.st_mtime_ns, stat.st_size))
                except OSError:
                    continue
        return tuple(sorted(entries))

    def detect_desktop_environment(self, refresh_env: bool = True) -> str:
        """Detects the currently running Desktop Environment or Window Manager."""
        if refresh_env:
            # Ensure we have refreshed display variables from systemd/sockets
            try:
                from src.tools.desktop import ensure_gui_environment
                ensure_gui_environment()
            except Exception:
                pass

        xdg_current = os.environ.get("XDG_CURRENT_DESKTOP", "").lower()
        desktop_session = os.environ.get("DESKTOP_SESSION", "").lower()
        gdm_session = os.environ.get("GDMSESSION", "").lower()

        # 1. Primary check: XDG_CURRENT_DESKTOP
        if "hyprland" in xdg_current:
            return "hyprland"
        elif "sway" in xdg_current:
            return "sway"
        elif "i3" in xdg_current:
            return "i3"
        elif "kde" in xdg_current or "plasma" in xdg_current:
            return "kde_plasma"
        elif "gnome" in xdg_current:
            return "gnome"
        elif "cosmic" in xdg_current:
            return "cosmic"
        elif "niri" in xdg_current:
            return "niri"

        # 2. Compositor / WM specific sockets and signatures
        if os.environ.get("HYPRLAND_INSTANCE_SIGNATURE"):
            return "hyprland"
        if os.environ.get("SWAYSOCK"):
            return "sway"
        if os.environ.get("I3SOCK"):
            return "i3"
        if os.environ.get("NIRI_SOCKET"):
            return "niri"
        if os.environ.get("KDE_SESSION_VERSION"):
            return "kde_plasma"

        # 3. Session variables
        if "hyprland" in desktop_session:
            return "hyprland"
        if "sway" in desktop_session:
            return "sway"
        if "i3" in desktop_session:
            return "i3"
        if "niri" in desktop_session:
            return "niri"
        if "kde" in desktop_session or "plasma" in desktop_session:
            return "kde_plasma"
        if "gnome" in desktop_session or "gnome" in gdm_session:
            return "gnome"
        if "cosmic" in desktop_session:
            return "cosmic"

        return "generic_desktop"

    def list_skills(self) -> List[Dict[str, str]]:
        """Lists all discovered skill files and identifies the active one."""
        active_de = self.detect_desktop_environment()
        discovered = {}

        # Search user directory first, then built-in
        for search_dir in reversed(self.get_skill_paths()):
            for file_path in search_dir.glob("*.md"):
                if file_path.name.lower() in ["readme.md", "skills.md"]:
                    continue
                skill_id = file_path.stem.replace(".skills", "").lower()
                discovered[skill_id] = {
                    "id": skill_id,
                    "filename": file_path.name,
                    "path": str(file_path),
                    "is_active": (skill_id == active_de),
                    "loaded_by_default": (skill_id == "computer_use" or skill_id == active_de),
                }

        return list(discovered.values())

    def load_skill(self, skill_name: str) -> Optional[str]:
        """Loads the content of a skill by name or ID."""
        clean_name = skill_name.strip().lower().replace(".md", "").replace(".skills", "")
        # Skill names are IDs, never paths. This keeps lookup inside the
        # configured skill directories even when a model supplies a bad name.
        if not clean_name or "/" in clean_name or "\\" in clean_name:
            return None
        # Aliases
        alias_map = {
            "plasma": "kde_plasma",
            "kde": "kde_plasma",
            "kwin": "kde_plasma",
            "swaywm": "sway",
            "i3wm": "i3",
            "gnome-shell": "gnome",
            "niri-wm": "niri",
            "generic": "generic_desktop",
            "default": "generic_desktop"
        }
        target_id = alias_map.get(clean_name, clean_name)

        possible_filenames = [
            f"{target_id}.md",
            f"{target_id}.skills.md",
            f"{clean_name}.md",
            f"{clean_name}.skills.md"
        ]

        for search_dir in self.get_skill_paths():
            for fname in possible_filenames:
                p = search_dir / fname
                if p.is_file():
                    try:
                        if not p.resolve().is_relative_to(search_dir.resolve()) or p.stat().st_size > 40_000:
                            continue
                        return p.read_text(encoding="utf-8")
                    except Exception:
                        pass
        return None

    def get_startup_context(self) -> str:
        """Load the core computer-use workflow and detected desktop skill."""
        general = self.load_skill("computer_use") or ""
        desktop = self.get_active_de_context()
        return (
            "=== CORE COMPUTER-USE SKILL (loaded at startup) ===\n"
            f"{general.strip()}\n"
            "=== END CORE COMPUTER-USE SKILL ===\n\n"
            f"{desktop}"
        )

    def get_active_de_context(self) -> str:
        """Retrieves formatted markdown context for the currently active desktop environment."""
        active_de = self.detect_desktop_environment()
        content = self.load_skill(active_de)
        if not content:
            content = self.load_skill("generic_desktop") or ""

        header = f"=== ACTIVE DESKTOP & WINDOW MANAGER SKILL ({active_de.upper()}) ==="
        footer = "=== END DESKTOP SKILL ==="
        return f"{header}\n{content.strip()}\n{footer}"

    def create_or_update_skill(
        self,
        skill_id: str,
        content: str,
        description: str = "",
        overwrite: bool = True,
    ) -> Path:
        """Saves or updates a skill in the user's custom skills directory."""
        import re
        clean_id = re.sub(r"[^a-zA-Z0-9_-]", "_", str(skill_id).strip().lower()).strip("_-")
        if not clean_id:
            raise ValueError(f"Invalid skill ID: {skill_id!r}")

        self.user_skills_dir.mkdir(parents=True, exist_ok=True)
        target_path = self.user_skills_dir / f"{clean_id}.md"

        if target_path.exists() and not overwrite:
            raise FileExistsError(f"Skill file {target_path} already exists and overwrite=False.")

        formatted_content = content.strip()
        if description and description.strip():
            desc_clean = description.strip()
            if not formatted_content.startswith("#"):
                title = clean_id.replace("_", " ").title()
                formatted_content = f"# Skill: {title}\n\n> {desc_clean}\n\n{formatted_content}"
            elif desc_clean not in formatted_content:
                lines = formatted_content.splitlines()
                title_line = lines[0]
                body_lines = lines[1:]
                formatted_content = f"{title_line}\n\n> {desc_clean}\n\n" + "\n".join(body_lines)

        target_path.write_text(formatted_content + "\n", encoding="utf-8")
        self._index_dirty = True
        return target_path

    def delete_skill(self, skill_id: str) -> bool:
        """Deletes a custom user skill by ID. Returns True if deleted."""
        clean_id = skill_id.strip().lower().replace(".md", "").replace(".skills", "")
        for fname in [f"{clean_id}.md", f"{clean_id}.skills.md"]:
            p = self.user_skills_dir / fname
            if p.is_file():
                try:
                    p.unlink()
                    self._index_dirty = True
                    return True
                except Exception:
                    pass
        return False

    def _build_index(self) -> None:
        """Builds in-memory lexical BM25 index over available non-default skills."""
        try:
            from src.memory.bm25 import BM25Index
        except Exception:
            class BM25Index:
                def __init__(self, *args, **kwargs):
                    self.doc_ids = []
                    self.texts = []
                def fit(self, doc_ids, texts):
                    self.doc_ids = list(doc_ids)
                    self.texts = list(texts)
                def score(self, query):
                    q_words = set(query.lower().split())
                    scores = []
                    for t in self.texts:
                        overlap = sum(1 for w in q_words if w in t.lower())
                        scores.append(overlap)
                    return scores

        active_de = self.detect_desktop_environment(refresh_env=False)
        all_skills = self.list_skills()
        self._indexed_skills.clear()

        doc_ids = []
        texts = []
        try:
            from src.memory.bm25 import tokenize
        except Exception:
            tokenize = lambda text: [word.lower() for word in re.findall(r"[a-zA-Z0-9]+", text)]
        for s in all_skills:
            s_id = s["id"]
            # Exclude skills already loaded by default at startup
            if s.get("loaded_by_default") or s_id in {"computer_use", active_de, "generic_desktop"}:
                continue
            content = self.load_skill(s_id)
            if not content:
                continue

            lines = [line.strip() for line in content.splitlines() if line.strip()]
            header = lines[0] if lines else ""
            desc = lines[1] if len(lines) > 1 else ""

            trigger_match = re.search(
                r"(?ims)^#{1,3}\s*when\s+to\s+use\s*$\n(.*?)(?=^#{1,3}\s|\Z)",
                content,
            )
            trigger_text = trigger_match.group(1)[:1200] if trigger_match else ""

            # Match using skill identity and its declared triggers, not procedure
            # text. Indexing full bodies made unrelated chat match incidental words
            # from long examples and caused thousands of irrelevant prompt tokens.
            index_text = (
                f"{s_id} {s_id.replace('_', ' ')} {header} {desc} {trigger_text}"
            )
            self._indexed_skills[s_id] = {
                "id": s_id,
                "header": header,
                "desc": desc,
                "match_tokens": set(tokenize(index_text)),
                "content": content,
                "path": s.get("path", ""),
            }
            doc_ids.append(s_id)
            texts.append(index_text)

        if doc_ids:
            self._bm25 = BM25Index()
            self._bm25.fit(doc_ids, texts)
        else:
            self._bm25 = None

        self._indexed_file_signature = self._skill_file_signature()
        self._index_dirty = False

    def match_skills(
        self,
        query: str,
        limit: int = 1,
        min_score: float = 4.0,
    ) -> List[Tuple[str, str]]:
        """Finds matching specialized skills for a user query.

        Returns a list of (skill_id, skill_content) tuples sorted by relevance.
        Excludes skills already loaded at startup (computer_use and active desktop).
        """
        if not query or not query.strip():
            return []

        if self._skill_file_signature() != self._indexed_file_signature:
            self._index_dirty = True
        if self._index_dirty or self._bm25 is None:
            self._build_index()

        if not self._indexed_skills or self._bm25 is None:
            return []

        clean_query = query.strip().lower()
        try:
            from src.memory.bm25 import tokenize
            query_tokens = set(tokenize(clean_query))
        except Exception:
            query_tokens = set(clean_query.split())

        if not query_tokens:
            return []

        try:
            bm25_scores = self._bm25.score(clean_query)
        except Exception:
            bm25_scores = [0.0] * len(self._indexed_skills)

        scored_candidates = []
        for (s_id, meta), bm25_s in zip(self._indexed_skills.items(), bm25_scores):
            score = float(bm25_s)
            id_tokens = set(s_id.replace("-", "_").split("_"))
            shared_id_tokens = query_tokens.intersection(id_tokens)
            if shared_id_tokens:
                score += 2.0 * len(shared_id_tokens)
            if s_id in clean_query or s_id.replace("_", " ") in clean_query:
                score += 3.0

            shared_query_tokens = query_tokens.intersection(meta.get("match_tokens", ()))
            # Skill IDs and descriptions often share broad words such as "work",
            # "email", or "desktop". A high BM25 score from one such word is not
            # enough to inject a multi-step procedure into an unrelated request.
            # Require two distinct terms for multiword requests, while keeping
            # short one-token prompts eligible for exact skill discovery.
            sufficiently_grounded = len(query_tokens) <= 1 or len(shared_query_tokens) >= 2
            if score >= min_score and sufficiently_grounded:
                scored_candidates.append((score, s_id, meta["content"]))

        scored_candidates.sort(key=lambda x: x[0], reverse=True)
        return [(s_id, content) for _, s_id, content in scored_candidates[:limit]]

    def get_matched_skill_context(self, query: str, max_chars: int = 4000) -> Optional[str]:
        """Returns formatted Markdown context for skills relevant to the query, or None."""
        matches = self.match_skills(query, limit=1)
        if not matches:
            return None
        skill_id, content = matches[0]
        excerpt = content[:max_chars].strip()
        if len(content) > max_chars:
            excerpt += "\n\n[Skill guidance truncated for context brevity.]"
        header = f"=== SPECIALIZED SKILL GUIDANCE ({skill_id.upper()}) ==="
        footer = "=== END SPECIALIZED SKILL GUIDANCE ==="
        return f"{header}\n{excerpt}\n{footer}"
