from __future__ import annotations
from geopy.geocoders import Nominatim
from timezonefinder import TimezoneFinder
from zoneinfo import ZoneInfo
import textwrap
import json
import os
import pickle
import re
import sqlite3
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from supabase import create_client, Client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

load_dotenv(
    Path(__file__).resolve().parent / ".env"
)

# ============================================================
# JARVIS
# LOCAL AI + FREE NEWS SEARCH + ONLINE RESEARCH
# ============================================================

APP_NAME = "SATURN"
VERSION = "3.0"
ASSISTANT = "PERSONAL AI ASSISTANT"

# ============================================================
# LOCAL BRAIN
# ============================================================

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
#OLLAMA_URL = "http://localhost:11434/api/generate"
LOCAL_MODEL = "qwen3:1.7b"

# ============================================================
# ONLINE BRAIN + SYSTEM
# ============================================================

GEMINI_MODEL = "openrouter/free"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_API_KEY = (
    os.getenv("OPENROUTER_API_KEY")
    or os.getenv("GEMINI_API_KEY")
)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SECRET_KEY = os.getenv(
    "SUPABASE_SECRET_KEY"
)
SUPABASE_PUBLISHABLE_KEY = os.getenv(
    "SUPABASE_PUBLISHABLE_KEY"
)

supabase: Client | None = None
SUPABASE_KEY_KIND = "none"

if SUPABASE_URL and (
    SUPABASE_SECRET_KEY
    or SUPABASE_PUBLISHABLE_KEY
):
    if SUPABASE_SECRET_KEY:
        _supabase_key = SUPABASE_SECRET_KEY
        SUPABASE_KEY_KIND = "secret"
    else:
        _supabase_key = SUPABASE_PUBLISHABLE_KEY
        SUPABASE_KEY_KIND = "publishable"

    try:
        supabase = create_client(
            SUPABASE_URL,
            _supabase_key
        )
    except Exception:
        supabase = None
# ============================================================
# FILES
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATABASE_PATH = BASE_DIR / "jarvis_memory.db"
CALENDAR_PATH = BASE_DIR / "calendar.json"
ML_MODEL_PATH = BASE_DIR / "saturn_intent_ml.pkl"
ML_DATA_PATH = BASE_DIR / "saturn_ml_data.json"

# ============================================================
# TIMEOUTS
# ============================================================

OLLAMA_TIMEOUT = 180
WEB_TIMEOUT = 20
GEMINI_TIMEOUT = 120

# ============================================================
# CONVERSATION
# ============================================================

MAX_HISTORY_MESSAGES = 12

# ============================================================
# MEMORY
# ============================================================

MAX_MEMORY_RESULTS = 6

# ============================================================
# NEWS / WEB RESEARCH LIMITS
# ============================================================

QUICK_SEARCH_RESULTS = 3
NORMAL_SEARCH_RESULTS = 5
DEEP_SEARCH_RESULTS = 8
VERY_DEEP_SEARCH_RESULTS = 15

QUICK_PAGES = 1
NORMAL_PAGES = 2
DEEP_PAGES = 5
VERY_DEEP_PAGES = 12

MAX_PAGE_CHARS = 7000
MAX_GEMINI_INPUT_CHARS = 24000
GENERAL_WEB_SEARCH_RESULTS = 8


# ============================================================
# JARVIS IDENTITY
# ============================================================

SYSTEM_PROMPT = """
You are SATURN, a personal AI assistant.

IDENTITY:
- Your name is SATURN.
- You were made by Uzair Abdullah.
- If asked who made you, who created you, who your maker is,
  or who your developer is, say:
  "I was made by Uzair Abdullah."
- If asked for an introduction, introduce yourself as SATURN,
  a personal AI assistant made by Uzair Abdullah.
  - If asked who you are, what you are, or who is SATURN,
  say that you are SATURN, a personal AI assistant made by
  Uzair Abdullah.
- Do not answer "I was made by Uzair Abdullah." when the user
  asks who you are. That answer is only for questions about
  who created or made you.
- Understand identity questions in English, Roman Urdu, Urdu,
  Hindi, and mixed language.
- Examples: "Who are you?", "tum kon ho?", "aap kon ho?",
  "آپ کون ہو؟", "اپ کون ہو؟" all mean the user is asking
  for SATURN's identity.

BEHAVIOR:
- Speak naturally and professionally.
- Use English by default.
- Be concise for simple questions.
- Be detailed when the user asks for detail.
- Never invent memories.
- Never invent calendar events.
- Never claim to have searched the web unless web search happened.
- Never claim to have read a webpage unless it was actually read.
- Never invent current information.
- When a live/news/web answer is requested, use OpenRouter's live web tools.
- Use web search and page fetching as evidence when needed.
- Clearly distinguish source information from your own reasoning.
- Always answer the user's CURRENT question.
- Do not repeat a previous answer unless the user asks you to repeat it.
- Do not introduce yourself unless the user asks who you are or asks for an introduction.
- Do not mention your creator unless the user asks about your creator.
- Do not reuse a previous response as the answer to a new question.
- Never respond with a generic greeting when the user asks a specific question.
- Treat each new user message as a new request unless it clearly refers to previous context.
- Reply only in English, Roman Urdu, or Urdu script.
- Do not write responses in Hindi/Devanagari script.
- You may understand Hindi or Hindi-script input, but reply in
  English, Roman Urdu, or Urdu script.
- For mixed English + Roman Urdu messages, prefer the same mixed style.
"""


# ============================================================
# TERMINAL DISPLAY
# ============================================================

class StatusDisplay:
    WIDTH = 78

    current_message: str = ""

    @classmethod
    def show(cls, message: str) -> None:
        cls.current_message = str(message or "").strip()
        print(
            f"\r{APP_NAME}: {message}",
            end="",
            flush=True
        )

    @classmethod
    def clear(cls) -> None:
        cls.current_message = ""
        print(
            "\r" + (" " * cls.WIDTH) + "\r",
            end="",
            flush=True
        )


def print_wrapped(
    text: str,
    width: int = 78,
    prefix: str = ""
) -> None:

    if not text:
        return

    paragraphs = text.split("\n")

    for paragraph in paragraphs:

        paragraph = paragraph.strip()

        if not paragraph:
            print()
            continue

        # URLs and long strings need special handling.
        lines = textwrap.wrap(
            paragraph,
            width=width,
            break_long_words=False,
            break_on_hyphens=False
        )

        if not lines:
            print()
            continue

        print(prefix + lines[0])

        for line in lines[1:]:
            print(" " * len(prefix) + line)

# ============================================================
# SUPABASE CONNECTION TEST
# ============================================================

def test_supabase_connection() -> bool:
    if supabase is None:
        return False

    try:
        supabase.table("memories").select(
            "id"
        ).limit(1).execute()

        supabase.table("calendar_events").select(
            "id"
        ).limit(1).execute()

        return True

    except Exception:
        return False

# ============================================================
# CALENDAR
# ============================================================

@dataclass
class CalendarEvent:
    title: str
    event_date: str
    time: str = ""
    priority: str = "normal"
    notes: str = ""


class CalendarStore:

    def __init__(self, path: Path):
        self.path = path
        self._ensure_file()

    def _ensure_file(self) -> None:

        if not self.path.exists():
            self.path.write_text(
                "[]",
                encoding="utf-8"
            )

    def events(self) -> list[CalendarEvent]:

        try:

            data = json.loads(
                self.path.read_text(
                    encoding="utf-8"
                )
            )

        except (OSError, json.JSONDecodeError):

            return []

        if not isinstance(data, list):
            return []

        events = []

        for item in data:

            if not isinstance(item, dict):
                continue

            title = str(
                item.get("title", "")
            ).strip()

            if not title:
                continue

            events.append(
                CalendarEvent(
                    title=title,
                    event_date=str(
                        item.get("date", "")
                    ),
                    time=str(
                        item.get("time", "")
                    ),
                    priority=str(
                        item.get("priority", "normal")
                    ),
                    notes=str(
                        item.get("notes", "")
                    )
                )
            )

        return events

    def today(self) -> list[CalendarEvent]:

        today = date.today().isoformat()

        return [
            event
            for event in self.events()
            if event.event_date == today
        ]

    def summary(self) -> str:

        today_display = date.today().strftime(
            "%A, %d %B %Y"
        )

        events = self.today()

        if not events:

            return (
                f"Today is {today_display}. "
                "There are no events in the local calendar."
            )

        lines = [
            f"Today: {today_display}",
            f"Total events: {len(events)}",
            "",
            "Schedule:"
        ]

        for number, event in enumerate(
            events,
            start=1
        ):

            time_text = (
                f" at {event.time}"
                if event.time
                else ""
            )

            priority_text = ""

            if event.priority.lower() != "normal":

                priority_text = (
                    f" [{event.priority}]"
                )

            lines.append(
                f"{number}. {event.title}"
                f"{time_text}"
                f"{priority_text}"
            )

        important = [
            event
            for event in events
            if event.priority.lower()
            in {"high", "important", "urgent"}
        ]

        if important:

            lines.append("")
            lines.append("Important:")

            for event in important:

                lines.append(
                    f"- {event.title}"
                )

        return "\n".join(lines)

    def add_event(self, event: CalendarEvent):

        events = self.events()

        # Prevent exact duplicate events.
        for existing in events:

            if (
                existing.title.strip().lower()
                == event.title.strip().lower()
                and existing.event_date == event.event_date
                and existing.time == event.time
            ):
                return False

        events.append(event)

        data = [
            {
                "title": item.title,
                "date": item.event_date,
                "time": item.time,
                "priority": item.priority,
                "notes": item.notes,
            }
            for item in events
        ]

        self.path.write_text(
            json.dumps(data, indent=4),
            encoding="utf-8"
        )

        return True

    def clear_all(self) -> None:

        self.path.write_text(
            "[]",
            encoding="utf-8"
        )


# ============================================================
# LONG-TERM MEMORY
# ============================================================

class MemoryStore:

    def __init__(self, path: Path):

        self.path = path
        self._initialize()

    def _initialize(self) -> None:

        with sqlite3.connect(
            self.path
        ) as db:

            db.execute(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL UNIQUE,
                    created_at TEXT NOT NULL
                )
                """
            )

    def save(
        self,
        content: str
    ) -> bool:

        content = content.strip()

        if not content:
            return False

        try:

            with sqlite3.connect(
                self.path
            ) as db:

                db.execute(
                    """
                    INSERT INTO memories
                    (content, created_at)
                    VALUES (?, ?)
                    """,
                    (
                        content,
                        datetime.now().isoformat(
                            timespec="seconds"
                        )
                    )
                )

            return True

        except sqlite3.IntegrityError:

            return False

    def all(self) -> list[str]:

        with sqlite3.connect(
            self.path
        ) as db:

            rows = db.execute(
                """
                SELECT content
                FROM memories
                ORDER BY id DESC
                """
            ).fetchall()

        return [
            row[0]
            for row in rows
        ]

    def search(
        self,
        query: str,
        limit: int = MAX_MEMORY_RESULTS
    ) -> list[str]:

        memories = self.all()

        if not memories:
            return []

        words = [
            re.sub(
                r"[^\w]",
                "",
                word.lower()
            )
            for word in query.split()
            if len(word) >= 3
        ]

        if not words:
            return memories[:limit]

        scored = []

        for memory in memories:

            lowered = memory.lower()

            score = sum(
                1
                for word in words
                if word in lowered
            )

            if score:
                scored.append(
                    (score, memory)
                )

        scored.sort(
            key=lambda item: item[0],
            reverse=True
        )

        return [
            memory
            for _, memory in scored[:limit]
        ]

    def clear(self) -> None:

        with sqlite3.connect(
            self.path
        ) as db:

            db.execute(
                "DELETE FROM memories"
            )


# ============================================================
# LOCAL QWEN BRAIN
# ============================================================

class LocalAI:

    def __init__(
        self,
        url: str,
        model: str
    ):

        self.url = url
        self.model = model

    def chat(
        self,
        messages: list[dict[str, str]]
    ) -> str:

        prompt_parts = []

        for message in messages:

            role = message.get(
                "role",
                "user"
            ).upper()

            content = message.get(
                "content",
                ""
            )

            prompt_parts.append(
                f"{role}:\n{content}"
            )

        prompt = "\n\n".join(
            prompt_parts
        )

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": True,
            "think": False
        }

        try:

            response = requests.post(
                self.url,
                json=payload,
                stream=True,
                timeout=OLLAMA_TIMEOUT
            )

            response.raise_for_status()

            StatusDisplay.show(
                "Thinking..."
            )

            full_response = ""

            for line in response.iter_lines():

                if not line:
                    continue

                try:

                    data = json.loads(
                        line
                    )

                except json.JSONDecodeError:

                    continue

                piece = data.get(
                    "response",
                    ""
                )

                if piece:
                    full_response += piece

            StatusDisplay.clear()

            answer = full_response.strip()

            if not answer:

                return (
                    "The local AI returned "
                    "an empty response."
                )

            return clean_ai_output(answer)

        except requests.exceptions.ConnectionError:

            StatusDisplay.clear()

            return (
                "I cannot connect to Ollama. "
                "Please make sure Ollama is running."
            )

        except requests.exceptions.Timeout:

            StatusDisplay.clear()

            return (
                "The local AI took too long to respond."
            )

        except requests.exceptions.HTTPError as error:

            StatusDisplay.clear()

            return f"Ollama HTTP error: {error}"

        except requests.exceptions.RequestException as error:

            StatusDisplay.clear()

            return f"Ollama connection error: {error}"

        except Exception as error:

            StatusDisplay.clear()

            return f"Local AI error: {error}"

    def intent(
        self,
        prompt: str
    ) -> dict[str, Any]:

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "format": "json"
        }

        try:
            StatusDisplay.show(
                "Thinking..."
            )

            response = requests.post(
                self.url,
                json=payload,
                timeout=60
            )

            response.raise_for_status()

            data = response.json()

            raw_response = data.get(
                "response",
                ""
            ).strip()

            if not raw_response:
                raise ValueError(
                    "Empty router response."
                )

            parsed = json.loads(raw_response)

            if not isinstance(
                parsed,
                dict
            ):
                raise ValueError(
                    "Router JSON is not an object."
                )

            StatusDisplay.clear()
            return parsed

        except requests.exceptions.RequestException as error:
            StatusDisplay.clear()
            raise RuntimeError(
                f"Router connection error: {error}"
            ) from error

        except json.JSONDecodeError as error:
            StatusDisplay.clear()
            raise RuntimeError(
                f"Router returned invalid JSON: {error}"
            ) from error

        except Exception:
            StatusDisplay.clear()
            raise
# ============================================================
# MACHINE LEARNING — TRAINING DATA
# ============================================================

class MLTrainingData:

    def __init__(self, path: Path):
        self.path = path
        self._ensure_file()

    def _ensure_file(self) -> None:
        if not self.path.exists():
            self.path.write_text(
                "[]",
                encoding="utf-8"
            )

    def load(self) -> list[dict[str, str]]:
        try:
            data = json.loads(
                self.path.read_text(
                    encoding="utf-8"
                )
            )

            if not isinstance(data, list):
                return []

            return [
                item
                for item in data
                if isinstance(item, dict)
                and item.get("text")
                and item.get("intent")
            ]

        except Exception:
            return []

    def add(
        self,
        text: str,
        intent: str
    ) -> None:

        text = str(text or "").strip()
        intent = str(intent or "").strip()

        if not text or not intent:
            return

        data = self.load()

        for item in data:
            if (
                item.get("text", "").strip().lower()
                == text.lower()
                and item.get("intent", "").strip().lower()
                == intent.lower()
            ):
                return

        data.append(
            {
                "text": text,
                "intent": intent
            }
        )

        self.path.write_text(
            json.dumps(
                data,
                indent=4,
                ensure_ascii=False
            ),
            encoding="utf-8"
        )
# ============================================================
# MACHINE LEARNING — LOCAL INTENT CLASSIFIER
# ============================================================

class MLIntentClassifier:

    TRAINING_DATA = (

        # LOCAL
        ("hello", "local"),
        ("how are you", "local"),
        ("what is python", "local"),
        ("explain gravity", "local"),
        ("what is the value of pi", "local"),
        ("help me understand this code", "local"),
        ("mujhe python samjhao", "local"),
        ("ye kya hai", "local"),

        # DATE
        ("what is today's date", "date"),
        ("what is the date today", "date"),
        ("aaj ki date kya hai", "date"),
        ("aaj ki tareekh kya hai", "date"),
        ("आज की तारीख क्या है", "date"),
        ("آج کی تاریخ کیا ہے", "date"),

        # TIME
        ("what time is it", "time"),
        ("what is the current time", "time"),
        ("abhi kitne bajay hain", "time"),
        ("abhi kya time hai", "time"),
        ("अभी कितने बजे हैं", "time"),
        ("ابھی کتنے بجے ہیں", "time"),

        # DATE + TIME
        ("what is the date and time", "date_time"),
        ("what are the date and time", "date_time"),
        ("aaj ki date aur time kya hai", "date_time"),
        ("aaj ki tareekh aur waqt kya hai", "date_time"),
        ("आज की तारीख और समय क्या है", "date_time"),
        ("آج کی تاریخ اور وقت کیا ہے", "date_time"),

        # WORLD TIME
        ("what time is it in london", "world_time"),
        ("time in london", "world_time"),
        ("what time is it in new york", "world_time"),
        ("london mein kitne bajay hain", "world_time"),
        ("paris mein kya time hai", "world_time"),
        ("لندن میں کتنے بجے ہیں", "world_time"),

        # NEWS
        ("latest news", "news"),
        ("latest openai news", "news"),
        ("what is the latest news", "news"),
        ("recent news about openai", "news"),
        ("openai ki latest news", "news"),
        ("aaj ki news kya hai", "news"),

        # WEB
        ("search the web for python", "web"),
        ("search the internet for qwen", "web"),
        ("look this up online", "web"),
        ("find this on the internet", "web"),
        ("python ke bare mein search karo", "web"),
        ("internet par ye search karo", "web"),

        # RESEARCH
        ("deep research on bears", "research"),
        ("research this deeply", "research"),
        ("do detailed research on this", "research"),
        ("compare different sources", "research"),
        ("is topic par deep research karo", "research"),
        ("multiple sources se research karo", "research"),

        # VERIFY
        ("check your previous answer", "verify"),
        ("verify your last answer", "verify"),
        ("double check your answer", "verify"),
        ("check that again", "verify"),
        ("mera pichla jawab dobara check karo", "verify"),
        ("phir se check karo", "verify"),

        # CALENDAR
        ("show my calendar", "calendar"),
        ("what events do I have", "calendar"),
        ("add an event for tomorrow", "calendar"),
        ("save an event for december", "calendar"),
        ("aaj kya events hain", "calendar"),
        ("mera calendar dikhao", "calendar"),
        ("12 december ko birthday hai", "calendar"),
        ("December mein kya events hain", "calendar"),
    )

    def __init__(self, model_path: Path):
        self.model_path = model_path
        self.vectorizer = None
        self.classifier = None
        self.enabled = False

        self._load_or_train()

    def _load_or_train(self) -> None:

        # Load an existing trained model.
        if self.model_path.exists():
            try:
                with self.model_path.open("rb") as file:
                    payload = pickle.load(file)

                self.vectorizer = payload["vectorizer"]
                self.classifier = payload["classifier"]
                self.enabled = True
                return

            except Exception:
                self.vectorizer = None
                self.classifier = None

        # Train a new model.
        try:
            texts = [
                text
                for text, _ in self.TRAINING_DATA
            ]

            labels = [
                label
                for _, label in self.TRAINING_DATA
            ]

            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                sublinear_tf=True
            )

            features = self.vectorizer.fit_transform(
                texts
            )

            self.classifier = LogisticRegression(
                max_iter=1000,
                random_state=42
            )

            self.classifier.fit(
                features,
                labels
            )

            with self.model_path.open("wb") as file:
                pickle.dump(
                    {
                        "vectorizer": self.vectorizer,
                        "classifier": self.classifier,
                    },
                    file
                )

            self.enabled = True

        except Exception:
            self.vectorizer = None
            self.classifier = None
            self.enabled = False

    def predict(
        self,
        text: str
    ) -> tuple[str, float]:

        if (
            not self.enabled
            or self.vectorizer is None
            or self.classifier is None
        ):
            return "local", 0.0

        try:
            features = self.vectorizer.transform(
                [str(text or "")]
            )

            probabilities = (
                self.classifier.predict_proba(
                    features
                )[0]
            )

            best_index = int(
                probabilities.argmax()
            )

            intent = str(
                self.classifier.classes_[best_index]
            )

            confidence = float(
                probabilities[best_index]
            )

            return intent, confidence

        except Exception:
            return "local", 0.0


# ============================================================
# ONLINE RESEARCHER — OPENROUTER WEB SEARCH + WEB FETCH
# ============================================================

class GeminiResearch:
    """
    Compatibility wrapper kept under the old class name so the rest
    of SATURN does not need to be rewired.

    All online search, webpage fetching, and online analysis are handled
    by OpenRouter through a single API request using server-side tools.
    """

    def __init__(self):

        self.enabled = bool(
            OPENROUTER_API_KEY
        )

        self.session = requests.Session()

        self.session.headers.update(
            {
                "Authorization":
                    f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type":
                    "application/json",
                "HTTP-Referer":
                    "http://localhost",
                "X-Title":
                    "SATURN Personal AI Assistant",
            }
        )

    def research(
        self,
        question: str,
        sources: list[dict[str, Any]],
        depth: str,
        search_kind: str = "web"
    ) -> str:

        if not self.enabled:
            return (
                "OpenRouter is not configured. "
                "Please check OPENROUTER_API_KEY."
            )

        if depth == "quick":
            depth_instruction = (
                "Give a short practical answer. "
                "Use web search only as much as needed."
            )
        elif depth == "deep":
            depth_instruction = (
                "Perform deeper web research. "
                "Use multiple searches or page fetches when useful. "
                "Compare relevant information and uncertainty."
            )
        elif depth == "very_deep":
            depth_instruction = (
                "Perform very deep web research. "
                "Use multiple searches and fetch relevant pages. "
                "Compare sources carefully and explain uncertainty."
            )
        else:
            depth_instruction = (
                "Give a useful answer based on current web information. "
                "Search or fetch pages when needed."
            )

        if search_kind == "news":
            request_type = (
                "This is a news request. Prioritize recent, relevant news "
                "and make dates clear."
            )
        else:
            request_type = (
                "This is a general web/research request. Find the most "
                "relevant and reliable current sources."
            )

        prompt = f"""
You are SATURN's online research analyst.

USER QUESTION:
{question}

REQUEST TYPE:
{search_kind}

{request_type}

RESEARCH DEPTH:
{depth}

INSTRUCTIONS:
{depth_instruction}

Rules:
- Use OpenRouter's live web_search tool when current or online information is needed.
- Use OpenRouter's web_fetch tool when reading a specific result/page would improve accuracy.
- You may perform multiple searches or page fetches when useful, especially for deep research.
- Prefer primary, official, and otherwise reliable sources.
- Cross-check important claims when practical.
- Do not invent facts, sources, dates, quotations, or URLs.
- Clearly mention uncertainty or conflicting information.
- Keep dates clear for current information.
- Answer the user's question directly.
- At the end provide a Sources section with useful source titles and URLs when web sources were used.
"""

        try:

            StatusDisplay.show(
                "Searching the web..."
                if search_kind != "news"
                else "Searching news..."
            )

            response = self.session.post(
                OPENROUTER_URL,
                json={
                    "model": GEMINI_MODEL,
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "tools": [
                        {
                            "type": "openrouter:web_search",
                            "parameters": {
                                "max_results": min(
                                    GENERAL_WEB_SEARCH_RESULTS,
                                    10
                                )
                            }
                        },
                        {
                            "type": "openrouter:web_fetch"
                        }
                    ],
                    "stream": False
                },
                timeout=GEMINI_TIMEOUT
            )

            StatusDisplay.clear()

            response.raise_for_status()

            data = response.json()

            choices = data.get(
                "choices",
                []
            )

            if not choices:
                return (
                    "OpenRouter returned no web-search answer."
                )

            message = choices[0].get(
                "message",
                {}
            )

            output = message.get(
                "content",
                ""
            )

            if isinstance(output, list):
                output = "".join(
                    str(item.get("text", ""))
                    if isinstance(item, dict)
                    else str(item)
                    for item in output
                )

            output = str(output).strip()

            if not output:
                return (
                    "OpenRouter returned an empty web-search answer."
                )

            return clean_ai_output(output)

        except requests.exceptions.Timeout:

            StatusDisplay.clear()

            return (
                "OpenRouter web search took too long."
            )

        except requests.exceptions.HTTPError as error:

            StatusDisplay.clear()

            try:
                details = response.json()
            except Exception:
                details = str(error)

            return (
                "OpenRouter web search failed."
                f"\n\nTechnical error: {details}"
            )

        except requests.exceptions.RequestException as error:

            StatusDisplay.clear()

            return (
                "OpenRouter web search connection failed."
                f"\n\nTechnical error: {error}"
            )

        except Exception as error:

            StatusDisplay.clear()

            return (
                "OpenRouter web search failed."
                f"\n\nTechnical error: {error}"
            )


# ============================================================
# RESEARCH ENGINE — OPENROUTER ONLY
# ============================================================

class ResearchEngine:

    def __init__(self):

        self.gemini = GeminiResearch()

    def research(
        self,
        query: str,
        depth: str = "normal",
        search_kind: str = "web"
    ) -> str:

        return self.gemini.research(
            query,
            [],
            depth,
            search_kind
        )


# ============================================================
# CLEAN AI OUTPUT
# ============================================================

def clean_ai_output(text: str) -> str:
    """
    Remove Markdown formatting so SATURN speaks/displays
    clean plain text in the terminal.
    """

    if not text:
        return ""

    # Remove headings such as ### Sources
    text = re.sub(
        r"^\s*#{1,6}\s*",
        "",
        text,
        flags=re.MULTILINE
    )

    # Remove bold/italic markers
    text = text.replace("**", "")
    text = text.replace("__", "")
    text = text.replace("*", "")

    # Remove Markdown code markers
    text = text.replace("```", "")
    text = text.replace("`", "")

    # Convert Markdown bullets into simple bullets
    text = re.sub(
        r"^\s*[-•]\s+",
        "- ",
        text,
        flags=re.MULTILINE
    )

    # Remove horizontal rules such as ---
    text = re.sub(
        r"^\s*-{3,}\s*$",
        "",
        text,
        flags=re.MULTILINE
    )

    # Clean excessive blank lines
    text = re.sub(
        r"\n{3,}",
        "\n\n",
        text
    )

    return text.strip()
# ============================================================
# JARVIS CORE
# ============================================================

class Jarvis:

    def __init__(self):

        self.ai = LocalAI(
            OLLAMA_URL,
            LOCAL_MODEL
        )

        self.memory = MemoryStore(
            DATABASE_PATH
        )

        self.calendar = CalendarStore(
            CALENDAR_PATH
        )

        self.researcher = ResearchEngine(

        )
        self.ml = MLIntentClassifier(
            ML_MODEL_PATH
        )
        self.ml_data = MLTrainingData(
            ML_DATA_PATH
        )

        self.geocoder = Nominatim(
            user_agent="jarvis-timezone-assistant"
        )

        self.timezone_finder = TimezoneFinder(
            in_memory=True
        )

        self.conversation: list[
            dict[str, str]
        ] = []

        self.last_research_query = ""

    # ========================================================
    # MEMORY
    # ========================================================

    @staticmethod
    def memory_requested(
        text: str
    ) -> bool:

        normalized = text.lower()

        triggers = (
            "remember this",
            "remember that",
            "remember it",
            "save this",
            "save that",
            "save it",
            "don't forget this",
            "don't forget that",
            "dont forget this",
            "dont forget that",
            "keep this in mind",
            "keep that in mind",
            "make a note of this",
            "make a note of that"
        )

        return any(
            trigger in normalized
            for trigger in triggers
        )

    def extract_memory(self) -> str:

        recent = self.conversation[-8:]

        transcript = "\n".join(
            f"{item['role']}: "
            f"{item['content']}"
            for item in recent
        )

        prompt = f"""
The user explicitly asked SATURN to remember
something.

Extract ONLY the useful long-term fact.

Rules:
- Return only the fact.
- Do not explain.
- Do not add information.
- Do not invent anything.

Conversation:
{transcript}
"""

        messages = [
            {
                "role": "system",
                "content":
                    "Extract useful long-term facts "
                    "for SATURN."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        return self.ai.chat(
            messages
        )
       

    # ========================================================
    # DATE / TIME
    # ========================================================

    @staticmethod
    def current_date_answer() -> str:
        now = datetime.now(ZoneInfo("Asia/Karachi"))

        return (
            f"Today is {now.strftime('%A, %d %B %Y')}."
        )

    @staticmethod
    def current_time_answer() -> str:
        now = datetime.now(ZoneInfo("Asia/Karachi"))

        return (
            f"The current time is {now.strftime('%H:%M')}."
        )

    # ========================================================
    # SEMANTIC INTENT ROUTER
    # ========================================================

    def detect_intent(
        self,
        user_message: str
    ) -> dict[str, Any]:
        ml_intent, ml_confidence = self.ml.predict(
    user_message
)

        router_prompt = f"""
You are SATURN's intent classifier.

Classify the CURRENT user message into exactly ONE intent:
local, date, time, date_time, world_time, news, web, research, verify, calendar

local:
Normal conversation, knowledge, math, explanations,
animals, science, coding, general questions, greetings,
casual conversation, checking, verifying, reviewing,
or instructions that do not explicitly ask for date/time.


date:
The user wants the current date.

time:
The user wants the current local time.
date_time:
The user wants both the current local date and current local time.
world_time:
The user wants the current time in one or more places.
news:
The user wants current news, headlines, recent events,
latest announcements, or breaking news.
web:
The user wants information searched on the general internet.
research:
The user wants deeper investigation, source comparison,
detailed research, or multi-source analysis.
verify:
The user wants SATURN to check, verify, double-check,
review, or re-evaluate a previous answer from the conversation.
The user expects SATURN to inspect its previous answer,
verify it independently, and report whether it was correct.
calendar:
The user wants to add, view, list, edit, or remove
calendar events.

IMPORTANT:
- Understand the meaning, not exact wording.
- The user may write in English, Hindi, Urdu,
  Roman Hindi, Roman Urdu, or mixed language.
- Identify the user's intent even when languages are mixed.
- Do not classify based only on English keywords.
- Translate the meaning internally before choosing the intent.
- Keep the original user question in the "query" field.
- Do not answer the user.
- Return JSON only.
- Do not add markdown.
- Do not invent information.

Examples:
"hello" -> local
"hello please check twice" -> verify
"please check this twice" -> verify
"double check your answer" -> verify
"verify this again" -> verify
"what is the date and time?" -> date_time
"What time is it?" -> time
"What is the date and time?" -> date_time
"aaj ki date aur time kya hai?" -> date_time
"आज की तारीख और समय क्या है?" -> date_time
"Time in London" -> world_time
"Latest OpenAI news" -> news
"Search the web for Python 3.14" -> web
"Deep research on bears" -> research
"What is the value of pi?" -> local
"abhi kitne bajay hain?" -> time
"london mein kitne bajay hain?" -> world_time
"openai ki latest news" -> news
"python 3.14 ke bare mein search karo" -> web
"bears par deep research karo" -> research
"ابھی کتنے بجے ہیں؟" -> time
"لندن میں کتنے بجے ہیں؟" -> world_time
"अभी कितने बजे हैं?" -> time
"लंदन में कितने बजे हैं?" -> world_time
"check that again" -> verify
"please check twice" -> verify
"double check your answer" -> verify
"verify your last answer" -> verify
"check your previous answer" -> verify
"dobara check karo" -> verify
"phir se check karo" -> verify
"add an event for 12 December 2026 someone's birthday" -> calendar
"events of december" -> calendar
"show my calendar" -> calendar
"aaj kya events hain?" -> calendar
"December mein kya events hain?" -> calendar
"12 December ko birthday hai" -> calendar
"12 December ka event save karo" -> calendar
Date:
"what is today's date?" -> date
"aaj ki tareekh kya hai?" -> date
"आज की तारीख क्या है?" -> date
"آج کی تاریخ کیا ہے؟" -> date

Date and time:
"what is the date and time?" -> date_time
"aaj ki date aur time kya hai?" -> date_time
"आज की तारीख और समय क्या है?" -> date_time
"آج کی تاریخ اور وقت کیا ہے؟" -> date_time

Verify:
"check your previous answer" -> verify
"mera pichla jawab dobara check karo" -> verify
"अपना पिछला जवाब दोबारा चेक करो" -> verify
"اپنا پچھلا جواب دوبارہ چیک کرو" -> verify

Return JSON only:
{{
  "intent": "local",
  "query": "original useful question",
  "locations": []
}}

LOCAL MACHINE-LEARNING HINT:
The local ML classifier predicts:
intent = {ml_intent}
confidence = {ml_confidence:.2f}

This ML result is only a hint.
Make the final classification yourself.

CURRENT USER MESSAGE:
{user_message}
"""

        try:
            result = self.ai.intent(
                router_prompt
            )

            if not isinstance(result, dict):
                raise ValueError(
                    "Router returned invalid data."
                )

            intent = result.get(
                "intent",
                "local"
            )

            if intent not in {
                "local",
                "date",
                "time",
                "date_time",
                "world_time",
                "news",
                "web",
                "research",
                "verify",
                "calendar",
            }:
                intent = "local"

            query = str(
                result.get(
                    "query",
                    user_message
                )
            ).strip()

            locations = result.get(
                "locations",
                []
            )

            if not isinstance(
                locations,
                list
            ):
                locations = []

            locations = [
                str(location).strip()
                for location in locations
                if str(location).strip()
            ]

            return {
                "intent": intent,
                "query": query or user_message,
                "locations": locations,
            }

        except Exception:
            calendar_fallback = self._calendar_fallback_intent(
                user_message
            )

            if calendar_fallback:
                return calendar_fallback

            if ml_confidence >= 0.70:
                return {
                    "intent": ml_intent,
                    "query": user_message,
                    "locations": [],
                }

            return {
                "intent": "local",
                "query": user_message,
                "locations": [],
            }

    # ========================================================
    # SEARCH INTENT
    # ========================================================

    @staticmethod
    def search_depth(
        text: str
    ) -> str:

        normalized = text.lower()

        very_deep_triggers = (
            "very deep",
            "extremely deep",
            "full investigation",
            "comprehensive research",
            "thorough investigation",
            "research thoroughly",
            "extreme deep"
        )

        deep_triggers = (
            "go deeper",
            "deep research",
            "deep research on",
            "do deep research",
            "research this deeply",
            "research this thoroughly",
            "look deeper",
            "dig deeper",
            "compare sources",
            "compare different sources",
            "detailed research",
            "investigate this"
        )

        quick_triggers = (
            "quick search",
            "quickly search",
            "just search",
            "quick lookup"
        )

        if any(
            trigger in normalized
            for trigger in very_deep_triggers
        ):

            return "very_deep"

        if any(
            trigger in normalized
            for trigger in deep_triggers
        ):

            return "deep"

        if any(
            trigger in normalized
            for trigger in quick_triggers
        ):

            return "quick"

        return "normal"

    @staticmethod
    def should_search_web(
        text: str
    ) -> bool:

        normalized = text.lower().strip()

        # Explicit commands
        if normalized.startswith("/search "):
            return True

        if normalized.startswith("/research "):
            return True

        # Deep research
        depth = Jarvis.search_depth(normalized)

        if depth in {"deep", "very_deep"}:
            return True

        # Current/news/event intent
        current_words = (
            "latest",
            "current news",
            "latest news",
            "recent news",
            "today's news",
            "todays news",
            "today news",
            "news",
            "headline",
            "headlines",
            "breaking news",
            "what happened today",
            "what happened",
            "what is happening",
            "what's happening",
            "recent",
            "today",
            "tonight",
            "this week",
            "this month's news",
            "announcement",
            "announcements",
            "release",
            "launched",
            "launches",
            "event",
            "events",
            "who won",
            "result",
            "results",
            "live score",
            "stock news",
            "market news",
        )

        if any(word in normalized for word in current_words):
            return True

        # Explicit online intent
        online_words = (
            "search the web",
            "search the internet",
            "search online",
            "look it up",
            "look this up",
            "google this",
            "find online",
            "find on the internet",
            "browse the web",
            "check online",
        )

        return any(word in normalized for word in online_words)

    @staticmethod
    def clean_search_query(text: str) -> str:
        """
        Clean only command/wrapper text.
        The semantic router supplies the actual search query.
        """
        text = str(text or "").strip()

        prefixes = (
            "/search ",
            "/research ",
        )

        lowered = text.lower()

        for prefix in prefixes:
            if lowered.startswith(prefix):
                return text[len(prefix):].strip()

        return text

        # Remove common conversational wrappers
        patterns = (
            r"^can you search (?:for )?",
            r"^please search (?:for )?",
            r"^search (?:for )?",
            r"^look up ",
            r"^find online ",
        )

        for pattern in patterns:

            text = re.sub(
                pattern,
                "",
                text,
                flags=re.IGNORECASE
            )

        return text.strip()
    # ========================================================
    # MEMORY CONTEXT
    # ========================================================

    def relevant_memories(
        self,
        text: str
    ) -> list[str]:

        results = self.memory.search(
            text
        )

        if results:

            return results

        return self.memory.all()[
            :5
        ]

    # ========================================================
    # SYSTEM PROMPT
    # ========================================================

    def build_system_prompt(
        self,
        user_message: str
    ) -> str:

        memories = self.relevant_memories(
            user_message
        )

        if memories:

            memory_text = "\n".join(
                f"- {memory}"
                for memory in memories
            )

        else:

            memory_text = (
                "No relevant long-term memory."
            )

        return (
            SYSTEM_PROMPT
            + "\n\n"
            + "RELEVANT LONG-TERM MEMORY:\n"
            + memory_text
            + "\n\n"
            + "TODAY'S LOCAL CALENDAR:\n"
            + self.calendar.summary()
        )

    # ========================================================
    # verify previous answer
    # ========================================================
    def verify_last_answer(self) -> str:

        previous_user = ""
        previous_assistant = ""

        for item in reversed(self.conversation[:-1]):

            if (
                    item["role"] == "assistant"
                    and not previous_assistant
            ):
                previous_assistant = item["content"]

            elif (
                    item["role"] == "user"
                    and previous_assistant
                    and not previous_user
            ):
                previous_user = item["content"]
                break

        if not previous_assistant:
            return (
                "There is no previous answer "
                "for me to check."
            )

        prompt = f"""
    You are verifying SATURN's previous answer.

    PREVIOUS USER QUESTION:
    {previous_user}

    PREVIOUS SATURN ANSWER:
    {previous_assistant}

    Check the previous answer independently.

     Rules:
    - Do not simply repeat the previous answer.
    - Recalculate or re-evaluate it yourself.
    - Say whether the previous answer was correct.
    - If it was wrong, give the corrected answer.
    - If it was correct, clearly say it was correct.
    - Only verify the exact calculation or question that was asked.
    - Do not introduce new calculations or operations that were not requested.
    - Do not add unnecessary division, multiplication, subtraction, or other operations.
    - Do not use LaTeX.
    - Do not use $ signs for mathematical expressions.
    - Write mathematical expressions in simple plain text.
    - Be concise.
    """

        messages = [
            {
                "role": "system",
                "content": (
                    "You are SATURN performing "
                    "a verification of its previous answer."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        return self.ai.chat(messages)
    # ========================================================
    # LOCAL RESPONSE
    # ========================================================

    def local_response(
            self,
            user_message: str
    ) -> str:

        recent_history = (
            self.conversation[
                -6:
            ]
        )

        history_text = "\n".join(
            f"{item['role'].upper()}: "
            f"{item['content']}"
            for item in recent_history
            if item["content"] != user_message
        )

        system_prompt = (
                self.build_system_prompt(
                    user_message
                )
                + "\n\n"
                + "IMPORTANT RESPONSE RULES:\n"
                + "- Answer ONLY the current user question.\n"
                + "- Do not repeat previous assistant answers.\n"
                + "- Do not give a generic greeting unless the user greeted you.\n"
                + "- Do not mention Uzair Abdullah unless asked.\n"
                + "- Do not introduce yourself unless asked.\n"
                + "- Do not change the subject.\n"
                + "- Keep simple factual answers direct.\n"
                + "- If the question is 'value of pi', answer the value of pi.\n"
                + "\nCURRENT USER QUESTION:\n"
                + user_message
        )

        messages = [
            {
                "role": "system",
                "content": system_prompt
            }
        ]

        if history_text:
            messages.append(
                {
                    "role": "user",
                    "content":
                        "Previous conversation context:\n"
                        + history_text
                }
            )

        messages.append(
            {
                "role": "user",
                "content":
                    "Answer this current question:\n"
                    + user_message
            }
        )

        return self.ai.chat(
            messages
        )

    # ========================================================
    # WORLD TIME
    # ========================================================

    def get_city_time(
            self,
            location_name: str
    ) -> str:

        try:
            StatusDisplay.show(
                f"Finding timezone for {location_name}..."
            )

            location = self.geocoder.geocode(
                location_name,
                exactly_one=True,
                timeout=10
            )

            StatusDisplay.clear()

            if location is None:
                return (
                    f"I couldn't find the location "
                    f"'{location_name}'."
                )

            timezone_name = (
                self.timezone_finder.timezone_at(
                    lng=location.longitude,
                    lat=location.latitude
                )
            )

            if not timezone_name:
                return (
                    f"I couldn't determine the timezone "
                    f"for {location_name}."
                )

            now = datetime.now(
                ZoneInfo(timezone_name)
            )

            return (
                f"{location_name.title()}: "
                f"{now.strftime('%H:%M')}"
            )

        except Exception as error:

            StatusDisplay.clear()

            return (
                f"{location_name.title()}: "
                f"unable to determine time."
            )

    def get_multiple_city_times(
            self,
            locations: list[str]
    ) -> str:

        results = []

        for location in locations:
            result = self.get_city_time(
                location
            )

            results.append(result)

        return "\n".join(results)

        # ========================================================
    # CALENDAR HELPERS
    # ========================================================

    @staticmethod
    def _calendar_month_number(text: str) -> int | None:

        months = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }

        lowered = text.lower()

        for name, number in months.items():
            if re.search(rf"\b{name}\b", lowered):
                return number

        return None

    def _calendar_fallback_intent(
        self,
        user_message: str
    ) -> dict[str, Any] | None:

        normalized = user_message.lower().strip()

        calendar_words = (
            "calendar",
            "event",
            "events",
            "birthday",
            "schedule",
            "reminder",
            "remind me",
        )

        month_names = (
            "january", "february", "march", "april",
            "may", "june", "july", "august",
            "september", "october", "november", "december"
        )

        if not any(
            word in normalized
            for word in calendar_words
        ) and not any(
            month in normalized
            for month in month_names
        ):
            return None

        list_markers = (
            "show",
            "list",
            "what events",
            "which events",
            "events of",
            "events in",
            "events for",
            "my calendar",
            "calendar events",
            "what is on my calendar",
            "what's on my calendar",
        )

        add_markers = (
            "add",
            "create",
            "save",
            "record",
            "schedule",
            "remind me",
        )

        if any(marker in normalized for marker in list_markers):
            action = "list"
        elif (
            self._calendar_month_number(normalized)
            and "event" in normalized
        ):
            action = "list"
        elif any(marker in normalized for marker in add_markers):
            action = "add"
        elif re.search(
            r"\b(event|birthday|reminder)\b.*\b\d{1,2}\b",
            normalized
        ):
            action = "add"
        else:
            return None

        return {
            "intent": "calendar",
            "query": user_message,
            "locations": [],
        }

    def _calendar_fallback_parse(
        self,
        user_message: str
    ) -> dict[str, str] | None:

        normalized = user_message.strip()
        lowered = normalized.lower()

        month_number = self._calendar_month_number(lowered)
        year_match = re.search(r"\b(20\d{2}|19\d{2})\b", lowered)
        year = year_match.group(1) if year_match else str(
            datetime.now(ZoneInfo("Asia/Karachi")).year
        )

        if month_number is None:
            return None

        date_match = re.search(
            r"\b(\d{1,2})(?:st|nd|rd|th)?\s+"
            r"(january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\b",
            lowered
        )

        if not date_match:
            date_match = re.search(
                r"\b(january|february|march|april|may|june|july|august|"
                r"september|october|november|december)\s+"
                r"(\d{1,2})(?:st|nd|rd|th)?\b",
                lowered
            )

        if not date_match:
            return None

        if date_match.group(1).isdigit():
            day = int(date_match.group(1))
        else:
            day = int(date_match.group(2))

        event_date = f"{int(year):04d}-{month_number:02d}-{day:02d}"

        try:
            datetime.strptime(event_date, "%Y-%m-%d")
        except ValueError:
            return None

        time_match = re.search(
            r"\b(?:at|@)\s+(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)\b",
            lowered
        )

        event_time = time_match.group(1) if time_match else ""

        title = normalized
        title = re.sub(
            r"^\s*(please\s+)?",
            "",
            title,
            flags=re.IGNORECASE
        )
        title = re.sub(
            r"\b(?:add|create|save|record|schedule|set)\b\s*",
            "",
            title,
            count=1,
            flags=re.IGNORECASE
        )
        title = re.sub(
            r"\b(?:an?|the)?\s*(?:calendar\s+)?(?:event|reminder)\b\s*",
            "",
            title,
            count=1,
            flags=re.IGNORECASE
        )
        title = re.sub(
            r"^\s*(?:for|on)\s+",
            "",
            title,
            flags=re.IGNORECASE
        )
        title = re.sub(
            r"\b(?:on|for)\s+\d{1,2}(?:st|nd|rd|th)?\s+"
            r"(?:january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\b",
            "",
            title,
            flags=re.IGNORECASE
        )
        title = re.sub(
            r"\b(?:on|for)\s+(?:january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\s+\d{1,2}(?:st|nd|rd|th)?\b",
            "",
            title,
            flags=re.IGNORECASE
        )
        title = re.sub(
            r"\b\d{1,2}(?:st|nd|rd|th)?\s+"
            r"(?:january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\b",
            "",
            title,
            flags=re.IGNORECASE
        )
        title = re.sub(
            r"\b(?:january|february|march|april|may|june|july|august|"
            r"september|october|november|december)\s+"
            r"\d{1,2}(?:st|nd|rd|th)?\b",
            "",
            title,
            flags=re.IGNORECASE
        )
        title = re.sub(
            r"\b(?:19|20)\d{2}\b",
            "",
            title
        )
        title = re.sub(
            r"^\s*(?:for|on)\s*$",
            "",
            title,
            flags=re.IGNORECASE
        )
        title = re.sub(
            r"\s+",
            " ",
            title
        ).strip(" ,.-")

        if not title:
            title = "Calendar Event"

        return {
            "title": title,
            "date": event_date,
            "time": event_time,
            "priority": "normal",
            "notes": "",
        }

    # ========================================================
    # CALENDAR REQUEST
    # ========================================================

    def calendar_request(self, user_message: str) -> str:

        prompt = f"""
    You are SATURN's calendar parser.

    Read the user's request and return JSON only.

    Format:
    {{
      "action": "add" or "list",
      "title": "",
      "date": "",
      "time": "",
      "priority": "normal",
      "notes": "",
      "month": "",
      "year": ""
    }}

    Rules:

    1. Use "add" ONLY when the user wants to create, save, add, or record an event.
    2. Use "list" when the user wants to see, show, list, or ask about calendar events.
    3. Examples of LIST:
       - "December events"
       - "show my calendar"
       - "what events are in December?"
       - "December mein kya events hain?"
       - "aaj kya events hain?"
    4. For a LIST request, never invent a date, title, or event.
    5. For ADD, date must be YYYY-MM-DD.
    6. If no time is given, use "".
    7. If no priority is given, use "normal".
    8. Keep the event title short and clear.
    9. Do not invent missing information.
    10. month should be a number from 1 to 12 when the user asks for a specific month.
    11. year should be four digits when given. If the user asks for a month but gives no year, leave year as "".

    USER REQUEST:
    {user_message}
    """

        try:
            result = self.ai.intent(prompt)

            if not isinstance(result, dict):
                raise ValueError("Calendar parser returned invalid data.")

        except Exception:
            fallback = self._calendar_fallback_parse(user_message)

            if fallback:
                result = {
                    "action": "add",
                    **fallback,
                }
            else:
                fallback_intent = self._calendar_fallback_intent(
                    user_message
                )

                if fallback_intent:
                    month = self._calendar_month_number(
                        user_message
                    )

                    events = self.calendar.events()

                    if month:
                        events = [
                            event
                            for event in events
                            if event.event_date[5:7] == f"{month:02d}"
                        ]

                    if not events:
                        if month:
                            month_name = datetime.strptime(
                                str(month),
                                "%m"
                            ).strftime("%B")
                            year = datetime.now(
                                ZoneInfo("Asia/Karachi")
                            ).year
                            return (
                                f"No saved events for "
                                f"{month_name} {year}."
                            )
                        return "Your calendar is empty."

                    lines = ["Calendar events:"]
                    for number, event in enumerate(events, start=1):
                        try:
                            date_text = datetime.strptime(
                                event.event_date,
                                "%Y-%m-%d"
                            ).strftime("%d %B %Y")
                        except ValueError:
                            date_text = event.event_date

                        time_text = (
                            f" at {event.time}"
                            if event.time
                            else ""
                        )

                        lines.append(
                            f"{number}. {date_text}{time_text} - {event.title}"
                        )

                    return "\n".join(lines)

                return "I couldn't understand that calendar request."

            action = str(result.get("action", "")).strip().lower()

            # ========================================================
            # LIST CALENDAR EVENTS
            # ========================================================

            if action == "list":

                month = str(result.get("month", "")).strip()
                year = str(result.get("year", "")).strip()

                events = self.calendar.events()

                if month:
                    try:
                        month_number = int(month)
                    except ValueError:
                        month_number = 0

                    if month_number:
                        events = [
                            event
                            for event in events
                            if int(event.event_date[5:7]) == month_number
                        ]

                if year:
                    events = [
                        event
                        for event in events
                        if event.event_date[:4] == year
                    ]

                if not events:
                    if month:
                        try:
                            month_name = datetime.strptime(
                                month,
                                "%m"
                            ).strftime("%B")
                        except ValueError:
                            month_name = "that month"

                        year_text = year if year else str(
                            datetime.now(
                                ZoneInfo("Asia/Karachi")
                            ).year
                        )

                        return (
                            f"No saved events for "
                            f"{month_name} {year_text}."
                        )

                    return "Your calendar is empty."

                lines = ["Calendar events:"]

                for number, event in enumerate(events, start=1):

                    try:
                        date_text = datetime.strptime(
                            event.event_date,
                            "%Y-%m-%d"
                        ).strftime("%d %B %Y")
                    except ValueError:
                        date_text = event.event_date

                    time_text = (
                        f" at {event.time}"
                        if event.time
                        else ""
                    )

                    lines.append(
                        f"{number}. "
                        f"{date_text}{time_text} - "
                        f"{event.title}"
                    )

                return "\n".join(lines)

            # ========================================================
            # ADD CALENDAR EVENT
            # ========================================================

            if action == "add":

                title = str(result.get("title", "")).strip()
                event_date = str(result.get("date", "")).strip()
                event_time = str(result.get("time", "")).strip()
                priority = str(
                    result.get("priority", "normal")
                ).strip() or "normal"
                notes = str(result.get("notes", "")).strip()

                if not title or not event_date:
                    return (
                        "I need the event name and date "
                        "before I can add it."
                    )

                try:
                    datetime.strptime(
                        event_date,
                        "%Y-%m-%d"
                    )
                except ValueError:
                    return "I couldn't understand the event date."

                event = CalendarEvent(
                    title=title,
                    event_date=event_date,
                    time=event_time,
                    priority=priority,
                    notes=notes,
                )

                saved = self.calendar.add_event(event)

                try:
                    display_date = datetime.strptime(
                        event_date,
                        "%Y-%m-%d"
                    ).strftime("%d %B %Y")
                except ValueError:
                    display_date = event_date

                time_text = (
                    f" at {event_time}"
                    if event_time
                    else ""
                )

                if not saved:
                    return (
                        f"That calendar event is already saved: "
                        f"{title} on {display_date}{time_text}."
                    )

                return (
                    f"Calendar event added: "
                    f"{title} on {display_date}{time_text}."
                )

            return "I couldn't understand that calendar request."

        except Exception:
            return "I couldn't process that calendar request."

    # ========================================================
    # MAIN RESPONSE
    # ========================================================

    def respond(
        self,
        user_message: str
    ) -> str:

        user_message = (
            user_message.strip()
        )

        self.conversation.append(
            {
                "role": "user",
                "content": user_message
            }
        )

        # ----------------------------------------------------
        # MEMORY
        # ----------------------------------------------------

        memory_notice = None

        if self.memory_requested(
            user_message
        ):
            StatusDisplay.show(
               "Saving to memory..."
        )
            memory = (
                self.extract_memory()
                .strip()
            )

            if (
                memory
                and not memory.startswith(
                    "I cannot connect"
                )
                and not memory.startswith(
                    "Local AI error"
                )
            ):

                if self.memory.save(
                    memory
                ):
                    memory_notice = "Memory saved."
                else:
                    memory_notice = (
                        "That memory is already saved."
                    )

        StatusDisplay.clear()
        # ----------------------------------------------------
        # CLEAR CALENDAR REQUEST
        # ----------------------------------------------------

        normalized = user_message.lower().strip()

        clear_calendar_phrases = (
            "clear my calendar",
            "clear the calendar",
            "clear calendar",
            "delete my calendar",
            "delete all calendar events",
            "delete all events",
            "remove all calendar events",
            "remove all events",
            "erase my calendar",
            "erase all calendar events",
        )

        if any(
            phrase in normalized
            for phrase in clear_calendar_phrases
        ):

            StatusDisplay.show(
                "Preparing calendar clear..."
            )
            StatusDisplay.clear()

            confirmation = input(
                f"{APP_NAME}: This will delete ALL calendar events. "
                "Are you sure? (yes/no): "
            ).strip().lower()

            if confirmation in {"yes", "y"}:
                self.calendar.clear_all()
                answer = "All calendar events have been cleared."
            else:
                answer = "Okay. The calendar was not changed."

            self.conversation.append(
                {
                    "role": "assistant",
                    "content": answer
                }
            )

            return answer

        # ----------------------------------------------------
        # SEMANTIC ROUTER
        # ----------------------------------------------------

        route = self.detect_intent(
            user_message
        )

        intent = route["intent"]
        query = route["query"]
        locations = route["locations"]
        # ----------------------------------------------------
        # VERIFY PREVIOUS ANSWER
        # ----------------------------------------------------

        if intent == "verify":
            answer = self.verify_last_answer()

        # ----------------------------------------------------
        # DATE
        # ----------------------------------------------------

        elif intent == "date":
            answer = self.current_date_answer()

        # ----------------------------------------------------
        # LOCAL TIME
        # ----------------------------------------------------

        elif intent == "time":
            answer = self.current_time_answer()
        # ----------------------------------------------------
        # time & date
        # ----------------------------------------------------
        elif intent == "date_time":
            answer = (
                    self.current_date_answer()
                    + "\n"
                    + self.current_time_answer()
            )
        # ----------------------------------------------------
        # calendar
        # ----------------------------------------------------
        elif intent == "calendar":

            StatusDisplay.show(
                "Opening calendar..."
            )

            answer = self.calendar_request(
                user_message
            )

            StatusDisplay.clear()
        # ----------------------------------------------------
        # WORLD TIME
        # ----------------------------------------------------

        elif intent == "world_time":

            if not locations:
                answer = (
                    "I couldn't determine the location "
                    "for that time request."
                )
            else:
                answer = self.get_multiple_city_times(
                    locations
                )

        # ----------------------------------------------------
        # NEWS / GENERAL WEB / RESEARCH
        # ----------------------------------------------------

        elif intent in {
            "news",
            "web",
            "research"
        }:

            depth = self.search_depth(
                user_message
            )

            if intent == "research":
                search_kind = "web"
            elif intent == "news":
                search_kind = "news"
            else:
                search_kind = "web"

            if (
                intent == "research"
                and query.lower().strip()
                in {
                    "go deeper",
                    "dig deeper",
                    "look deeper",
                    "do deep research",
                    "deep research"
                }
                and self.last_research_query
            ):
                query = self.last_research_query

            self.last_research_query = query

            answer = self.researcher.research(
                query,
                depth,
                search_kind
            )

        # ----------------------------------------------------
        # LOCAL AI
        # ----------------------------------------------------

        else:
            answer = self.local_response(
                user_message
            )

        # ----------------------------------------------------
        # SAVE ASSISTANT RESPONSE
        # ----------------------------------------------------

        self.conversation.append(
            {
                "role": "assistant",
                "content": answer
            }
        )

        if len(
            self.conversation
        ) > MAX_HISTORY_MESSAGES:

            self.conversation = (
                self.conversation[
                    -MAX_HISTORY_MESSAGES:
                ]
            )

        answer = clean_ai_output(
            answer
        )

        if memory_notice:
            return (
                f"{memory_notice}\n\n"
                f"{answer}"
            )

        return answer


# ============================================================
# TERMINAL UI
# ============================================================

def print_header() -> None:

    print()

    print("=" * 76)

    print(
        f"{APP_NAME:^72}"
    )

    print(
        f"{ASSISTANT:^72}"
    )

    print(
        f"{VERSION:^73}"
    )

    print("=" * 76)

    print()

    print(
        "Local brain  : Qwen 3 1.7B"
    )

    print(
        "Online brain : OpenRouter Free"
    )

    print(
        "Web search   : OpenRouter web tools"
    )

    print()


# ============================================================
# HELP
# ============================================================

def print_help() -> None:

    print()
    print(f"{APP_NAME} commands:")
    print("  /help          Show this help")
    print("  /memory        Show long-term memory")
    print("  /clear         Clear long-term memory")
    print("  /today         Show today's plan")
    print("  /calendar      Show all calendar events")
    print("  /clear_calendar Clear all calendar events")
    print("  /search X      Search the web")
    print("  /research X    Research a topic")
    print("  /exit          Exit SATURN")
    print()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    jarvis = Jarvis()

    print_header()

    while True:

        try:

            user_input = input(
                "Uzair: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print(
                f"\n{APP_NAME}: Shutting down."
            )

            break

        if not user_input:
            continue

        command = user_input.lower()

        # ----------------------------------------------------
        # HELP
        # ----------------------------------------------------

        if command == "/help":

            print_help()
            continue

        # ----------------------------------------------------
        # EXIT
        # ----------------------------------------------------

        if command in {
            "/exit",
            "exit",
            "quit"


        }:

            print(
                f"{APP_NAME}: Shutting down."
            )

            break

        # ----------------------------------------------------
        # MEMORY
        # ----------------------------------------------------

        if command == "/memory":

            memories = jarvis.memory.all()

            print()

            if not memories:

                print(
                    f"{APP_NAME}: Long-term memory "
                    "is empty."
                )

            else:

                print(
                    f"{APP_NAME}: Long-term memory:"
                )

                for number, memory in enumerate(
                    memories,
                    start=1
                ):

                    print(
                        f"{number}. {memory}"
                    )

            print()

            continue

        # ----------------------------------------------------
        # CLEAR MEMORY
        # ----------------------------------------------------

        if command == "/clear":

            jarvis.memory.clear()

            print(
                f"{APP_NAME}: Long-term memory cleared."
            )

            print()

            continue

        # ----------------------------------------------------
        # TODAY
        # ----------------------------------------------------

        if command == "/today":

            print()

            print(
                f"{APP_NAME}:"
            )

            print(
                jarvis.calendar.summary()
            )

            print()

            continue

        # ----------------------------------------------------
        # CALENDAR
        # ----------------------------------------------------

        if command == "/calendar":

            StatusDisplay.show(
                "Opening calendar..."
            )

            events = jarvis.calendar.events()

            StatusDisplay.clear()

            print()

            if not events:

                print(
                    f"{APP_NAME}: Calendar is empty."
                )

            else:

                print(
                    f"{APP_NAME}: Calendar events:"
                )

                for number, event in enumerate(
                    events,
                    start=1
                ):

                    try:
                        date_text = datetime.strptime(
                            event.event_date,
                            "%Y-%m-%d"
                        ).strftime("%d %B %Y")
                    except ValueError:
                        date_text = event.event_date

                    time_text = (
                        f" at {event.time}"
                        if event.time
                        else ""
                    )

                    print(
                        f"{number}. "
                        f"{date_text}{time_text} - "
                        f"{event.title}"
                    )

            print()

            continue

        # ----------------------------------------------------
        # CLEAR CALENDAR
        # ----------------------------------------------------

        if command == "/clear_calendar":

            print()

            confirmation = input(
                f"{APP_NAME}: This will delete ALL calendar events. "
                "Are you sure? (yes/no): "
            ).strip().lower()

            if confirmation in {"yes", "y"}:
                jarvis.calendar.clear_all()
                print(
                    f"{APP_NAME}: All calendar events cleared."
                )
            else:
                print(
                    f"{APP_NAME}: Calendar was not changed."
                )

            print()

            continue

        # ----------------------------------------------------
        # NORMAL / ONLINE RESPONSE
        # ----------------------------------------------------

        answer = jarvis.respond(
            user_input
        )

        print()
        print_wrapped(
            answer,
            width=78,
            prefix=f"{APP_NAME}: "
        )
        print()


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    print()
    print("Checking Supabase connection...")

    if test_supabase_connection():
        print("✅ Supabase is working.")
    else:
        print("❌ Supabase is NOT working.")

    print()

    main()
