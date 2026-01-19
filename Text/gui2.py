# who_is_spy_gui_debug.py
# -*- coding: utf-8 -*-
"""
Human vs AI "Who is Spy" (Tkinter GUI) —— with runtime logs + background thread for model calls (avoid UI freezing)

Key changes (per your latest requirements):
1) Within a word pair: user plays at most 2 rounds (one for each word)
   - First: user randomly gets one word (maintain your original "random" logic)
   - Second: if the other word hasn't been described, force assign the "other word" to user (avoid repeating same word 3~5 times)
   - If user gets spy word second time: user is spy; otherwise randomly pick another player as spy
2) Resume priority: assign word first, then check "if user word already described"
   - If assigned word already described: immediately skip to the other undescribed word; if both described skip the pair
3) File writing: player_inputs.json / who_is_spy_log.txt / runtime_debug.log all written to exe directory (exe_dir_path)
   - Resource reading (csv) still uses resource_path

Dependencies:
- Your project must have model/<model_name>/load.py implementing:
    play_round(player, history, log, field, language, is_trick) -> bool
    voting_round(player, history, log, language) -> dict  (at least containing {player_id: target_id})
"""

import csv
import importlib
import json
import os
import random
import sys
import time
import traceback
import threading
import queue
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional

import tkinter as tk
from tkinter import ttk, messagebox


# ---------------------------
# Utilities (logging / IO)
# ---------------------------
def resource_path(relative: str) -> str:
    """
    Read resources: from sys._MEIPASS when packaged; from current directory during development
    """
    if hasattr(sys, "_MEIPASS"):
        return os.path.join(sys._MEIPASS, relative)
    return os.path.join(os.path.abspath("."), relative)


def exe_dir_path(filename: str) -> str:
    """
    Write user data to:
    - When packaged: exe directory
    - During development: current working directory (can be changed to project root as needed)
    """
    if getattr(sys, "frozen", False):
        base_dir = os.path.dirname(sys.executable)
    else:
        base_dir = os.path.abspath(".")
    return os.path.join(base_dir, filename)


def save_log_to_file(log_lines: List[str], path: str) -> None:
    if not log_lines:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for line in log_lines:
            f.write(line + "\n")


def log_with_timestamp(log: List[str], msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log.append(f"[{ts}] {msg}")


def safe_json_load(path: str, default):
    if not os.path.exists(path):
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def safe_json_dump(path: str, obj) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def is_quota_error(e: Exception) -> bool:
    msg = str(e).lower()
    keywords = [
        "quota", "insufficient", "exceeded", "rate limit", "429",
        "balance", "limit", "quota", "usage", "rate limit", "throttling", "exceed",
        "api_key client option must be set", "openaierror"
    ]
    return any(k in msg for k in keywords)


def read_word_pairs(csv_file: str) -> List[Tuple[str, str]]:
    """
    Read csv word pairs: default uses first two columns as (civilian_word, spy_word).
    Simple heuristic to skip header.
    """
    pairs: List[Tuple[str, str]] = []
    with open(csv_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        rows = list(reader)

    if not rows:
        return pairs

    def looks_like_header(r: List[str]) -> bool:
        if len(r) < 2:
            return True
        a, b = (r[0] or "").strip(), (r[1] or "").strip()
        header_tokens = ["civilian", "spy", "平民", "卧底", "word", "词"]
        s = (a + " " + b).lower()
        return any(t in s for t in header_tokens)

    start_idx = 1 if looks_like_header(rows[0]) else 0
    for r in rows[start_idx:]:
        if len(r) < 2:
            continue
        cw = (r[0] or "").strip()
        sw = (r[1] or "").strip()
        if cw and sw:
            pairs.append((cw, sw))
    return pairs


class DebugLogger:
    """
    Runtime debug log: write to file + maintain tail buffer for GUI display
    """
    def __init__(self, path: str, max_tail: int = 400):
        self.path = path
        self.max_tail = max_tail
        self._lock = threading.Lock()
        self._tail: List[str] = []
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

    def write(self, msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        with self._lock:
            self._tail.append(line)
            if len(self._tail) > self.max_tail:
                self._tail = self._tail[-self.max_tail:]
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def tail_text(self) -> str:
        with self._lock:
            return "\n".join(self._tail)


# ---------------------------
# Game data structures
# ---------------------------
@dataclass
class Player:
    player_id: str
    word: str
    model: str

    def __str__(self) -> str:
        return f"{self.player_id}(model={self.model}, word={self.word})"


def get_model_api(model_name: str):
    """
    Consistent with Aihumanv2.py: dynamically import model.<model_name>.load
    """
    try:
        model_module = importlib.import_module(f"model.{model_name}.load")
        return model_module
    except ModuleNotFoundError:
        raise Exception(f"Model {model_name} API module does not exist, please confirm model name is correct")


# ---------------------------
# GUI App
# ---------------------------
class WhoIsSpyApp(tk.Tk):
    def __init__(
            self,
            csv_file: str,
            model_names: List[str],
            language: str = "cn",
            is_trick: bool = True,
            player_inputs_path: str = "player_inputs.json",
            log_file_path: str = "who_is_spy_log.txt",
            debug_log_path: str = "runtime_debug.log",
            seed: int = 42,
            word_num: int = 0,
    ):
        super().__init__()
        self.title("Who is Spy - Human vs AI (with runtime logs)")
        self.geometry("980x600")

        random.seed(seed)

        self.csv_file = csv_file
        self.model_names = model_names[:]
        if "user" not in self.model_names:
            raise ValueError("model_names must contain 'user' for the human player.")

        self.num_players = len(self.model_names)
        self.language = language
        self.is_trick = is_trick
        self.player_inputs_path = player_inputs_path
        self.log_file_path = log_file_path

        self.debug = DebugLogger(debug_log_path, max_tail=500)
        self.debug.write("=== App start ===")

        self.field = self._infer_field(csv_file)
        self.word_num = word_num

        # Worker queue for inter-thread communication
        self._worker_queue = queue.Queue()
        self._ui_poll_ms = 150
        self.after(self._ui_poll_ms, self._poll_worker_queue)

        # Read word pairs
        all_pairs = read_word_pairs(csv_file)
        if word_num and word_num > 0:
            all_pairs = all_pairs[:word_num]
        self.all_pairs = all_pairs
        self.debug.write(f"Loaded word pairs: {len(self.all_pairs)} from {csv_file}")

        # Player input file (resume play)
        self.player_inputs = safe_json_load(
            self.player_inputs_path,
            {"common": {self.language: {}}}
        )
        self.player_inputs.setdefault("common", {})
        self.player_inputs["common"].setdefault(self.language, {})
        self.user_desc_map: Dict[str, str] = self.player_inputs["common"][self.language]
        self.debug.write(f"Loaded player inputs: {len(self.user_desc_map)} words described")

        # Statistics structures
        self.all_player_stats: Dict[str, Dict[str, int]] = {}
        self.model_stats: Dict[str, Dict[str, int]] = {}

        # Initialize players
        self.players: List[Player] = [
            Player(player_id=f"player{i+1}", word="", model=self.model_names[i])
            for i in range(self.num_players)
        ]
        for p in self.players:
            self.model_stats[p.model] = self.model_stats.get(p.model, {
                "civilian_count": 0,
                "spy_count": 0,
                "correct_votes": 0,
                "spy_wins": 0,
                "illegial_times": 0
            })

        self.user_player: Player = next(p for p in self.players if p.model == "user")

        # Per-round state
        self.current_pair_index: int = -1
        self.current_civilian_word: str = ""
        self.current_spy_word: str = ""
        self.current_spy_player: Optional[Player] = None
        self.history: List[Dict[str, str]] = []

        # Round log buffer
        self.round_log_buf: List[str] = []

        # Frame container
        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True)

        self.frame_round = RoundFrame(self.container, self)
        self.frame_vote = VoteFrame(self.container, self)
        self.frame_result = ResultFrame(self.container, self)

        for fr in (self.frame_round, self.frame_vote, self.frame_result):
            fr.grid(row=0, column=0, sticky="nsew")

        self._prepare_next_round_or_finish()

    def _infer_field(self, csv_file: str) -> str:
        base = os.path.splitext(os.path.basename(csv_file))[0]
        parts = base.split("_")
        if len(parts) >= 2:
            return parts[1]
        return "default"

    def _system_prompt(self) -> str:
        if self.language == "cn":
            return ("你正在游玩·谁是卧底·的游戏。每轮游戏会有一个词汇对，大多数人会得到其中一个词汇，"
                    "他们被称为‘平民’，有1个人会获得另一个词汇，他被称为‘卧底’。平民的任务是通过描述和推理找出唯一的卧底，"
                    "卧底的任务是隐瞒自己与其他人不同的词语，混淆视听避免被发现。")
        return ("You are playing the game 'Who is Spy'. In each round, there is a pair of words. "
                "Most players will receive one of the words and are called 'civilians'. One player will receive the other word and is called the 'spy'. "
                "The civilians' task is to identify the spy through descriptions and reasoning, while the spy's task is to conceal their different word and avoid being discovered.")

    def _pair_needs_user_input(self, cw: str, sw: str) -> bool:
        # Both words described -> skip; otherwise continue
        return (cw not in self.user_desc_map) or (sw not in self.user_desc_map)

    def _find_next_pair_index(self) -> Optional[int]:
        for i, (cw, sw) in enumerate(self.all_pairs):
            if self._pair_needs_user_input(cw, sw):
                return i
        return None

    def _pick_user_word_for_pair(self, cw: str, sw: str) -> Optional[str]:
        """
        Choose which word user must get within a pair:
        - cw described, sw not described -> user must get sw
        - sw described, cw not described -> user must get cw
        - both undescribed -> user randomly gets one (first random)
        - both described -> None (skip entire pair)
        """
        cw_done = cw in self.user_desc_map
        sw_done = sw in self.user_desc_map

        if cw_done and sw_done:
            return None
        if cw_done and (not sw_done):
            return sw
        if sw_done and (not cw_done):
            return cw
        return random.choice([cw, sw])

    def _safe_flush_round_log(self) -> None:
        if self.round_log_buf:
            save_log_to_file(self.round_log_buf, self.log_file_path)
            self.round_log_buf = []

    def _safe_save_all(self) -> None:
        self.player_inputs["common"][self.language] = self.user_desc_map
        safe_json_dump(self.player_inputs_path, self.player_inputs)
        self._safe_flush_round_log()

    def _hard_stop_with_save(self, reason: str, exc: Optional[Exception] = None) -> None:
        try:
            self.debug.write(f"[STOP] {reason}")
            if exc is not None:
                self.debug.write(f"[STOP] Exception: {exc}")
                self.debug.write(traceback.format_exc())

            log_with_timestamp(self.round_log_buf, f"[STOP] {reason}")
            if exc is not None:
                log_with_timestamp(self.round_log_buf, f"[STOP] Exception: {exc}")

            self._safe_save_all()
        finally:
            messagebox.showerror(
                "Stopped and Saved",
                f"{reason}\n\nSaved to:\n- {self.player_inputs_path}\n- {self.log_file_path}\n- {os.path.basename(self.debug.path)}"
            )
            self.destroy()
            sys.exit(0)

    def _show_frame(self, frame: ttk.Frame) -> None:
        frame.tkraise()

    # ---------------------------
    # Worker queue polling (UI thread)
    # ---------------------------
    def _poll_worker_queue(self) -> None:
        try:
            while True:
                msg = self._worker_queue.get_nowait()
                kind = msg.get("kind")

                if kind == "goto_vote":
                    dialogue = msg.get("dialogue", "")
                    self.frame_vote.load_dialogue(dialogue)
                    self._show_frame(self.frame_vote)

                elif kind == "goto_result":
                    payload = msg["payload"]
                    self.frame_result.load_result(**payload)
                    self._show_frame(self.frame_result)

                elif kind == "error_stop":
                    reason = msg.get("reason", "Background thread stopped abnormally")
                    exc_text = msg.get("exc_text", "")
                    self._hard_stop_with_save(f"{reason}\n\n{exc_text}")

        except queue.Empty:
            pass

        self.after(self._ui_poll_ms, self._poll_worker_queue)

    # ---------------------------
    # Round lifecycle
    # ---------------------------
    def _prepare_next_round_or_finish(self) -> None:
        nxt = self._find_next_pair_index()
        if nxt is None:
            self._finalize_and_show_finish()
            return

        self.current_pair_index = nxt
        self.current_civilian_word, self.current_spy_word = self.all_pairs[nxt]
        self.debug.write(f"=== New round idx={nxt} cw={self.current_civilian_word} sw={self.current_spy_word} ===")

        # ---------------------------
        # Assign words (new logic)
        # ---------------------------
        user_word = self._pick_user_word_for_pair(self.current_civilian_word, self.current_spy_word)

        # Defense: if both words already described, skip this pair
        if user_word is None:
            self.debug.write(f"[ASSIGN] pair already completed, skip idx={nxt}")
            # Find next pair directly
            self._prepare_next_round_or_finish()
            return

        # Resume priority: assign first, then check; if already described, force switch to other
        if user_word in self.user_desc_map:
            self.debug.write(f"[ASSIGN] user_word '{user_word}' already described, switch to other word")
            other = self.current_spy_word if user_word == self.current_civilian_word else self.current_civilian_word
            if other in self.user_desc_map:
                self.debug.write(f"[ASSIGN] both words already described after re-check, skip pair idx={nxt}")
                self._prepare_next_round_or_finish()
                return
            user_word = other

        # Set all players to civilian word by default
        for p in self.players:
            p.word = self.current_civilian_word

        # Determine spy based on user_word
        if user_word == self.current_spy_word:
            # User is spy
            self.user_player.word = self.current_spy_word
            self.current_spy_player = self.user_player
            self.debug.write(f"[ASSIGN] user is SPY, user_word={user_word}")
        else:
            # User is civilian, randomly select another player as spy
            self.user_player.word = self.current_civilian_word
            non_user = [p for p in self.players if p.model != "user"]
            self.current_spy_player = random.choice(non_user)
            self.current_spy_player.word = self.current_spy_word
            self.debug.write(f"[ASSIGN] user is CIVILIAN, spy={self.current_spy_player.player_id}/{self.current_spy_player.model}")

        self.debug.write(f"[ROUND] idx={nxt} user_word={self.user_player.word} spy_player={self.current_spy_player.player_id}")

        # Initialize history/log
        self.history = [{"role": "system", "content": self._system_prompt()}]
        self.round_log_buf = []
        log_with_timestamp(self.round_log_buf, "------ Game Start ------")
        log_with_timestamp(self.round_log_buf, f"Word Pair: Civilian={self.current_civilian_word} / Spy={self.current_spy_word}")
        log_with_timestamp(self.round_log_buf, "Player assignments:")
        for p in self.players:
            log_with_timestamp(self.round_log_buf, str(p))
        self._safe_flush_round_log()

        # Round frame
        self.frame_round.load_round(
            round_index=nxt + 1,
            your_word=self.user_player.word,
            prefill=self.user_desc_map.get(self.user_player.word, "")
        )
        self._show_frame(self.frame_round)

    # ---------------------------
    # Step 1: user description
    # ---------------------------
    def on_user_submit_description(self, desc: str) -> None:
        desc = (desc or "").strip()
        if not desc:
            messagebox.showwarning("Description Required", "Please enter a description of your word before continuing.")
            return

        # surrogate check
        for ch in desc:
            if 0xD800 <= ord(ch) <= 0xDFFF:
                messagebox.showerror("Invalid Character", "Input contains invalid Unicode characters (surrogate). Please modify and resubmit.")
                return

        # Save player input
        self.user_desc_map[self.user_player.word] = desc
        self._safe_save_all()

        # Write to history + log
        self.history.append({"role": self.user_player.player_id, "content": desc})
        self.round_log_buf = []
        log_with_timestamp(self.round_log_buf, f"{self.user_player.player_id} generated valid output: {desc}")
        self._safe_flush_round_log()

        self.debug.write("[UI] User submitted description, entering model speech phase (background thread)")

        # Disable button to prevent duplicate clicks
        self.frame_round.set_busy(True)

        # Start background thread for model speech
        t = threading.Thread(target=self._worker_models_play_round, daemon=True)
        t.start()

    def _visible_dialogue_text_for_user(self) -> str:
        visible = [e for e in self.history if e["role"] not in ("system", self.user_player.player_id)]
        lines = [f"{e['role']}: {e['content']}" for e in visible]
        return "\n".join(lines) if lines else "(No other player descriptions yet)"

    def _worker_models_play_round(self) -> None:
        """
        Background thread: call each model's play_round sequentially, record time and exceptions.
        On success, notify main thread to jump to voting page.
        """
        try:
            players_order = self.players[:]
            random.shuffle(players_order)
            self.debug.write(f"[WORKER] Model speech order: {[p.player_id + ':' + p.model for p in players_order]}")

            for p in players_order:
                if p.model == "user":
                    continue

                self.debug.write(f"[WORKER] -> Starting model speech: {p.player_id} / model={p.model}")

                attempt = 0
                while True:
                    attempt += 1
                    start = time.time()
                    try:
                        self.debug.write(f"[WORKER]    get_model_api({p.model})")
                        model_api = get_model_api(p.model)

                        tmp_log: List[str] = []
                        self.debug.write(f"[WORKER]    play_round attempt={attempt}")
                        flag = model_api.play_round(p, self.history, tmp_log, self.field, self.language, self.is_trick)

                        if tmp_log:
                            save_log_to_file(tmp_log, self.log_file_path)

                        dt = time.time() - start
                        self.debug.write(f"[WORKER]    play_round done flag={flag} cost={dt:.3f}s")

                        if not flag:
                            self.model_stats[p.model]["illegial_times"] += 1
                            self.debug.write(f"[WORKER]    Invalid output -> illegial_times[{p.model}]={self.model_stats[p.model]['illegial_times']}, retry")
                            continue

                        break

                    except Exception as e:
                        dt = time.time() - start
                        exc_text = traceback.format_exc()
                        self.debug.write(f"[WORKER]    play_round exception cost={dt:.3f}s err={e}")
                        self.debug.write(exc_text)

                        if is_quota_error(e):
                            self._worker_queue.put({
                                "kind": "error_stop",
                                "reason": "Detected API quota/limit issue (play_round phase), stopped and saved.",
                                "exc_text": str(e)
                            })
                            return

                        self.debug.write("[WORKER]    Retry this model play_round after 20 seconds")
                        time.sleep(20)

            self.debug.write("[WORKER] All model speeches completed, preparing to enter voting page")

            dialogue = self._visible_dialogue_text_for_user()
            self._worker_queue.put({"kind": "goto_vote", "dialogue": dialogue})

        except Exception as e:
            exc_text = traceback.format_exc()
            self.debug.write(f"[WORKER] Background thread general exception: {e}")
            self.debug.write(exc_text)
            self._worker_queue.put({
                "kind": "error_stop",
                "reason": "Uncaught exception in background thread (models_play_round)",
                "exc_text": exc_text
            })
        finally:
            self.after(0, lambda: self.frame_round.set_busy(False))

    # ---------------------------
    # Step 2: user vote
    # ---------------------------
    def on_user_submit_vote(self, target_player_id: str) -> None:
        if not target_player_id:
            messagebox.showwarning("Vote Required", "Please select a player to vote out.")
            return

        self.debug.write(f"[UI] User submitted vote {target_player_id}, entering voting phase (background thread)")
        self.frame_vote.set_busy(True)

        t = threading.Thread(target=self._worker_vote_and_resolve, args=(target_player_id,), daemon=True)
        t.start()

    def _worker_vote_and_resolve(self, user_target: str) -> None:
        """
        Background thread: model voting_round + resolution + return result page payload
        """
        try:
            all_vote_details: Dict[str, str] = {}
            all_vote_details[self.user_player.player_id] = user_target

            # Write round log
            buf: List[str] = []
            log_with_timestamp(buf, f"{self.user_player.player_id} voted for {user_target}")
            save_log_to_file(buf, self.log_file_path)

            for p in self.players:
                if p.model == "user":
                    continue
                self.debug.write(f"[VOTE] -> voting_round {p.player_id}/{p.model}")

                start = time.time()
                try:
                    model_api = get_model_api(p.model)
                    tmp_log: List[str] = []
                    vote_details = model_api.voting_round(p, self.history, tmp_log, self.language)
                    if tmp_log:
                        save_log_to_file(tmp_log, self.log_file_path)

                    dt = time.time() - start
                    self.debug.write(f"[VOTE] <- voting_round {p.player_id}/{p.model} cost={dt:.3f}s ret={vote_details}")

                    all_vote_details[p.player_id] = vote_details.get(p.player_id, "")

                except Exception as e:
                    dt = time.time() - start
                    self.debug.write(f"[VOTE] voting_round exception cost={dt:.3f}s err={e}")
                    self.debug.write(traceback.format_exc())

                    if is_quota_error(e):
                        self._worker_queue.put({
                            "kind": "error_stop",
                            "reason": "Detected API quota/limit issue (voting_round phase), stopped and saved.",
                            "exc_text": str(e)
                        })
                        return

                    self._worker_queue.put({
                        "kind": "error_stop",
                        "reason": "Unrecoverable exception in model voting phase, stopped and saved.",
                        "exc_text": str(e)
                    })
                    return

            payload = self._resolve_round_build_payload(all_vote_details)
            self._worker_queue.put({"kind": "goto_result", "payload": payload})

        except Exception as e:
            exc_text = traceback.format_exc()
            self.debug.write(f"[WORKER] vote_and_resolve general exception: {e}")
            self.debug.write(exc_text)
            self._worker_queue.put({
                "kind": "error_stop",
                "reason": "Uncaught exception in background thread (vote_and_resolve)",
                "exc_text": exc_text
            })
        finally:
            self.after(0, lambda: self.frame_vote.set_busy(False))

    def _compute_model_rates_view(self) -> Dict[str, Dict[str, float]]:
        view = {}
        for model, stats in self.model_stats.items():
            civilian_hit_rate = (stats["correct_votes"] / stats["civilian_count"]) if stats["civilian_count"] > 0 else 0.0
            spy_win_rate = (stats["spy_wins"] / stats["spy_count"]) if stats["spy_count"] > 0 else 0.0
            view[model] = {
                "civilian_hit_rate": civilian_hit_rate,
                "spy_win_rate": spy_win_rate,
                "illegial_times": float(stats.get("illegial_times", 0)),
                "civilian_count": float(stats["civilian_count"]),
                "spy_count": float(stats["spy_count"]),
            }
        return view

    def _resolve_round_build_payload(self, all_vote_details: Dict[str, str]) -> Dict:
        """
        Resolution logic (consistent with Aihumanv2.py statistics), returns payload needed for result page.
        """
        spy_player = self.current_spy_player
        assert spy_player is not None

        # Count votes
        total_votes = {p.player_id: 0 for p in self.players}
        for voter_id, target in all_vote_details.items():
            if target in total_votes:
                total_votes[target] += 1

        max_votes = max(total_votes.values()) if total_votes else 0
        most_voted = [k for k, v in total_votes.items() if v == max_votes] if total_votes else []
        elimination = random.choice(most_voted) if most_voted else random.choice([p.player_id for p in self.players])

        if elimination == spy_player.player_id:
            result = "Spy failed, civilians win!"
            civilian_win = True
        else:
            result = "Civilians failed, spy wins!"
            civilian_win = False

        # Initialize player statistics
        for p in self.players:
            if p.player_id not in self.all_player_stats:
                self.all_player_stats[p.player_id] = {
                    "vote_correct_count": 0,
                    "is_spy_count": 0,
                    "spy_win_count": 0,
                    "is_civilian_count": 0,
                    "civilian_win_count": 0,
                }

        # Identity statistics
        for p in self.players:
            if p == spy_player:
                self.all_player_stats[p.player_id]["is_spy_count"] += 1
                if not civilian_win:
                    self.all_player_stats[p.player_id]["spy_win_count"] += 1
            else:
                self.all_player_stats[p.player_id]["is_civilian_count"] += 1
                if civilian_win:
                    self.all_player_stats[p.player_id]["civilian_win_count"] += 1

        # Civilians only hit if they vote for spy
        for voter_id, voted_id in all_vote_details.items():
            voter = next(pp for pp in self.players if pp.player_id == voter_id)
            if voter != spy_player and voted_id == spy_player.player_id:
                self.all_player_stats[voter_id]["vote_correct_count"] += 1

        # Incremental statistics by model
        for p in self.players:
            m = p.model
            if p == spy_player:
                self.model_stats[m]["spy_count"] += 1
                if not civilian_win:
                    self.model_stats[m]["spy_wins"] += 1
            else:
                self.model_stats[m]["civilian_count"] += 1

        for voter_id, voted_id in all_vote_details.items():
            voter = next(pp for pp in self.players if pp.player_id == voter_id)
            if voter != spy_player and voted_id == spy_player.player_id:
                self.model_stats[voter.model]["correct_votes"] += 1

        # Write round log
        buf: List[str] = []
        log_with_timestamp(buf, f"Player {elimination} eliminated")
        log_with_timestamp(buf, result)
        log_with_timestamp(buf, "------ Game End ------")
        save_log_to_file(buf, self.log_file_path)

        # Save player input and logs
        self._safe_save_all()

        user_hit = (self.user_player != spy_player and all_vote_details.get(self.user_player.player_id) == spy_player.player_id)

        payload = dict(
            round_index=self.current_pair_index + 1,
            civilian_word=self.current_civilian_word,
            spy_word=self.current_spy_word,
            spy_player_id=spy_player.player_id,
            elimination_id=elimination,
            result_text=result,
            user_vote=all_vote_details.get(self.user_player.player_id, ""),
            user_hit=user_hit,
            all_votes=all_vote_details,
            per_player_stats=self.all_player_stats,
            per_model_stats=self._compute_model_rates_view()
        )

        self.debug.write(f"[RESOLVE] elimination={elimination} spy={spy_player.player_id} civilian_win={civilian_win}")

        return payload

    # ---------------------------
    # Step 3: next round
    # ---------------------------
    def on_next_round(self) -> None:
        self.debug.write("[UI] Entering next round")
        self._prepare_next_round_or_finish()

    def _finalize_and_show_finish(self) -> None:
        # End: output final statistics to who_is_spy_log.txt
        buf: List[str] = []
        log_with_timestamp(buf, "------ Game Statistics ------")
        for pid, st in self.all_player_stats.items():
            log_with_timestamp(buf, f"{pid} detailed stats:")
            log_with_timestamp(buf, f"  - Times assigned as spy: {st['is_spy_count']}")
            log_with_timestamp(buf, f"  - Spy wins: {st['spy_win_count']}")
            log_with_timestamp(buf, f"  - Times assigned as civilian: {st['is_civilian_count']}")
            log_with_timestamp(buf, f"  - Civilian wins: {st['civilian_win_count']}")
            log_with_timestamp(buf, f"  - Civilian vote hits: {st['vote_correct_count']}")

        log_with_timestamp(buf, "------ Statistics by Model ------")
        view = self._compute_model_rates_view()
        for model, v in view.items():
            log_with_timestamp(buf, f"Model {model}:")
            log_with_timestamp(buf, f"  - Civilian vote hit rate: {v['civilian_hit_rate']:.4f}")
            log_with_timestamp(buf, f"  - Spy win rate: {v['spy_win_rate']:.4f}")
            log_with_timestamp(buf, f"  - Illegal output count: {int(v['illegial_times'])}")

        save_log_to_file(buf, self.log_file_path)
        self._safe_save_all()

        self.debug.write("=== All rounds finished ===")

        messagebox.showinfo(
            "All Completed",
            f"All word pairs completed.\n\nSaved to:\n- {self.player_inputs_path}\n- {self.log_file_path}\n- {os.path.basename(self.debug.path)}"
        )
        self.destroy()
        sys.exit(0)


# ---------------------------
# Frames
# ---------------------------
class RoundFrame(ttk.Frame):
    def __init__(self, parent, app: WhoIsSpyApp):
        super().__init__(parent)
        self.app = app

        self.lbl_title = ttk.Label(self, text="Step 1: Describe Your Word", font=("Arial", 16, "bold"))
        self.lbl_title.pack(anchor="w", padx=12, pady=(12, 2))

        self.lbl_round = ttk.Label(self, text="Current Round: -", font=("Arial", 11))
        self.lbl_round.pack(anchor="w", padx=12, pady=(0, 6))

        self.lbl_word = ttk.Label(self, text="Your Word: ", font=("Arial", 13))
        self.lbl_word.pack(anchor="w", padx=12, pady=(0, 6))

        self.txt_desc = tk.Text(self, height=9, wrap="word", font=("Arial", 12))
        self.txt_desc.pack(fill="x", padx=12, pady=6)

        self.btn_next = ttk.Button(self, text="Next Step: Generate and Enter Voting", command=self._on_next)
        self.btn_next.pack(anchor="e", padx=12, pady=8)

        self.note = ttk.Label(self, text="After submission, each model will generate descriptions sequentially (executed in background thread, will not freeze).", foreground="#444")
        self.note.pack(anchor="w", padx=12, pady=(0, 6))

    def load_round(self, round_index: int, your_word: str, prefill: str = "") -> None:
        self.lbl_round.config(text=f"Current Round: {round_index}")
        self.lbl_word.config(text=f"Your Word: {your_word}")
        self.txt_desc.delete("1.0", "end")
        if prefill:
            self.txt_desc.insert("1.0", prefill)

    def set_busy(self, busy: bool) -> None:
        self.btn_next.configure(state=("disabled" if busy else "normal"))

    def _on_next(self) -> None:
        desc = self.txt_desc.get("1.0", "end").strip()
        self.app.on_user_submit_description(desc)


class VoteFrame(ttk.Frame):
    def __init__(self, parent, app: WhoIsSpyApp):
        super().__init__(parent)
        self.app = app

        self.lbl_title = ttk.Label(self, text="Step 2: View All Descriptions and Vote", font=("Arial", 16, "bold"))
        self.lbl_title.pack(anchor="w", padx=12, pady=(12, 6))

        self.txt_dialogue = tk.Text(self, height=16, wrap="word", font=("Consolas", 11))
        self.txt_dialogue.pack(fill="both", expand=True, padx=12, pady=6)
        self.txt_dialogue.configure(state="disabled")

        bottom = ttk.Frame(self)
        bottom.pack(fill="x", padx=12, pady=10)

        ttk.Label(bottom, text="Select a player to eliminate:").pack(side="left")
        self.vote_var = tk.StringVar(value="")
        self.cmb = ttk.Combobox(bottom, textvariable=self.vote_var, state="readonly", width=20)
        self.cmb.pack(side="left", padx=8)

        self.btn_next = ttk.Button(bottom, text="Next Step: Resolve This Round", command=self._on_next)
        self.btn_next.pack(side="right")

    def load_dialogue(self, dialogue_text: str) -> None:
        self.txt_dialogue.configure(state="normal")
        self.txt_dialogue.delete("1.0", "end")
        self.txt_dialogue.insert("1.0", dialogue_text)
        self.txt_dialogue.configure(state="disabled")

        ids = [p.player_id for p in self.app.players]
        self.cmb["values"] = ids
        self.vote_var.set(ids[0] if ids else "")

    def set_busy(self, busy: bool) -> None:
        self.btn_next.configure(state=("disabled" if busy else "normal"))

    def _on_next(self) -> None:
        self.app.on_user_submit_vote(self.vote_var.get().strip())


class ResultFrame(ttk.Frame):
    def __init__(self, parent, app: WhoIsSpyApp):
        super().__init__(parent)
        self.app = app

        self.lbl_title = ttk.Label(self, text="Step 3: This Round's Result", font=("Arial", 16, "bold"))
        self.lbl_title.pack(anchor="w", padx=12, pady=(12, 6))

        self.txt_result = tk.Text(self, height=18, wrap="word", font=("Arial", 11))
        self.txt_result.pack(fill="both", expand=True, padx=12, pady=6)
        self.txt_result.configure(state="disabled")

        self.btn_next = ttk.Button(self, text="Enter Next Round", command=self.app.on_next_round)
        self.btn_next.pack(anchor="e", padx=12, pady=10)

    def load_result(
            self,
            round_index: int,
            civilian_word: str,
            spy_word: str,
            spy_player_id: str,
            elimination_id: str,
            result_text: str,
            user_vote: str,
            user_hit: bool,
            all_votes: Dict[str, str],
            per_player_stats: Dict[str, Dict[str, int]],
            per_model_stats: Dict[str, Dict[str, float]]
    ) -> None:
        lines: List[str] = []
        lines.append(f"Round: {round_index}")
        lines.append(f"Word Pair: Civilian = {civilian_word}; Spy = {spy_word}")
        lines.append(f"Spy Player: {spy_player_id}")
        lines.append(f"Eliminated Player: {elimination_id}")
        lines.append(f"This Round Result: {result_text}")
        lines.append("")
        lines.append(f"Your Vote: {user_vote}; Hit Spy: {'Yes' if user_hit else 'No'}")
        lines.append("")
        lines.append("All Votes:")
        for k, v in all_votes.items():
            lines.append(f"  - {k} -> {v}")
        lines.append("")
        lines.append("Cumulative Stats (by Player):")
        for pid, st in per_player_stats.items():
            lines.append(
                f"{pid}: vote_correct={st['vote_correct_count']}, spy={st['is_spy_count']}, spy_win={st['spy_win_count']}, "
                f"civilian={st['is_civilian_count']}, civilian_win={st['civilian_win_count']}"
            )
        lines.append("")
        lines.append("Cumulative Stats (by Model):")
        for m, v in per_model_stats.items():
            lines.append(
                f"{m}: civilian vote hit rate={v['civilian_hit_rate']:.4f}, spy win rate={v['spy_win_rate']:.4f}, "
                f"illegal output count={int(v['illegial_times'])}, civilian_count={int(v['civilian_count'])}, spy_count={int(v['spy_count'])}"
            )

        text = "\n".join(lines)
        self.txt_result.configure(state="normal")
        self.txt_result.delete("1.0", "end")
        self.txt_result.insert("1.0", text)
        self.txt_result.configure(state="disabled")


# ---------------------------
# Main
# ---------------------------
def main():
    # Read resources (csv) using resource_path
    csv_file = resource_path("data/word_日常_cn.csv")

    # Must contain user
    model_names = ["deepseek", "glm", "Qwen", "gpt", "ernie", "user"]
    language = "cn"
    is_trick = True

    # Write user data/logs: use exe_dir_path (fixed to exe directory)
    player_inputs_path = exe_dir_path("player_inputs.json")
    log_file_path = exe_dir_path("who_is_spy_log.txt")
    debug_log_path = exe_dir_path("runtime_debug.log")

    # word_num=0 means run entire csv; >0 means only run first N pairs
    word_num = 0

    app = WhoIsSpyApp(
        csv_file=csv_file,
        model_names=model_names,
        language=language,
        is_trick=is_trick,
        player_inputs_path=player_inputs_path,
        log_file_path=log_file_path,
        debug_log_path=debug_log_path,
        seed=42,
        word_num=word_num
    )
    app.mainloop()


if __name__ == "__main__":
    main()