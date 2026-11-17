#!/usr/bin/env python3
"""Convert DeepSeek conversations.json exports into readable UTF-8 text."""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, TextIO


VERSION = "1.1.0"
INVALID_WINDOWS_FILENAME = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
MAX_TITLE_CHARS = 40
SAFE_WINDOWS_PATH_LENGTH = 240
SPEAKER_BY_FRAGMENT = {
    "REQUEST": "我",
    "RESPONSE": "AI",
}
SPEAKER_BY_ROLE = {
    "user": "我",
    "human": "我",
    "assistant": "AI",
    "ai": "AI",
    "model": "AI",
}


class ConverterError(Exception):
    """An expected, user-facing conversion error."""


def filesystem_path(path: Path) -> Path:
    """Return an extended Windows path when the normal path is too long."""
    if os.name != "nt":
        return path

    absolute = str(path.resolve())
    if len(absolute) < SAFE_WINDOWS_PATH_LENGTH or absolute.startswith("\\\\?\\"):
        return path
    if absolute.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + absolute[2:])
    return Path("\\\\?\\" + absolute)


def flatten_content(value: Any) -> str:
    """Convert strings or nested content structures into readable text."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        parts = [flatten_content(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in ("text", "content", "parts", "value", "title", "name"):
            if key in value:
                text = flatten_content(value[key])
                if text:
                    return text
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def normalize_content(text: str) -> str:
    """Normalize newlines while preserving code blocks and lists."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text.strip("\n")


def to_single_line(text: str) -> str:
    """Collapse text for titles and filenames."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t\f\v]*\n[ \t\f\v]*", " ", text)
    text = re.sub(r"[ \t\f\v]+", " ", text)
    return text.strip()


def append_visible(
    messages: list[tuple[str, str]], speaker: str, text: str
) -> None:
    content = normalize_content(text)
    if not content.strip():
        return
    if messages and messages[-1][0] == speaker:
        messages[-1] = (speaker, messages[-1][1] + "\n" + content)
    else:
        messages.append((speaker, content))


def extract_message_texts(message: dict[str, Any]) -> list[tuple[str, str]]:
    """Extract visible messages from fragments, with role/content fallback."""
    messages: list[tuple[str, str]] = []
    fragments = message.get("fragments")
    if isinstance(fragments, list):
        for fragment in fragments:
            if not isinstance(fragment, dict):
                continue
            speaker = SPEAKER_BY_FRAGMENT.get(str(fragment.get("type", "")))
            if speaker is None:
                continue
            content = flatten_content(fragment.get("content"))
            append_visible(messages, speaker, content)

    if messages:
        return messages

    role = message.get("role")
    author = message.get("author")
    if not role and isinstance(author, dict):
        role = author.get("role")
    speaker = SPEAKER_BY_ROLE.get(str(role or "").lower())
    if speaker is None:
        return []

    content = message.get("content")
    if content is None:
        content = message.get("text")
    append_visible(messages, speaker, flatten_content(content))
    return messages


def parse_timestamp(value: Any) -> float | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    except (ValueError, OSError, OverflowError):
        return None


def choose_leaf(mapping: dict[str, Any]) -> str | None:
    """Choose the most likely current leaf when current_node is absent."""
    child_ids: set[str] = set()
    for node in mapping.values():
        if not isinstance(node, dict):
            continue
        children = node.get("children")
        if isinstance(children, list):
            child_ids.update(child for child in children if isinstance(child, str))

    leaves = [node_id for node_id in mapping if node_id not in child_ids]
    if not leaves:
        leaves = list(mapping)

    best_id: str | None = None
    best_key: tuple[bool, float, int] | None = None
    for position, node_id in enumerate(leaves):
        node = mapping.get(node_id)
        timestamp: float | None = None
        if isinstance(node, dict):
            message = node.get("message")
            if isinstance(message, dict):
                timestamp = parse_timestamp(message.get("inserted_at"))
        key = (timestamp is not None, timestamp or 0.0, position)
        if best_key is None or key > best_key:
            best_key = key
            best_id = node_id
    return best_id


def ordered_mapping_nodes(conversation: dict[str, Any]) -> list[dict[str, Any]]:
    """Follow current_node parents, or preserve mapping order as a fallback."""
    mapping = conversation.get("mapping")
    if not isinstance(mapping, dict):
        return []

    node_items = [
        (node_id, node)
        for node_id, node in mapping.items()
        if isinstance(node_id, str) and isinstance(node, dict)
    ]
    if not node_items:
        return []

    has_parent_links = any(
        isinstance(node.get("parent"), str) for _, node in node_items
    )
    if not has_parent_links:
        return [node for _, node in node_items]

    current = conversation.get("current_node")
    if isinstance(current, dict):
        current = current.get("id")
    if not isinstance(current, str) or current not in mapping:
        current = choose_leaf(mapping)

    chain: list[dict[str, Any]] = []
    seen: set[str] = set()
    while isinstance(current, str) and current in mapping and current not in seen:
        seen.add(current)
        node = mapping[current]
        if not isinstance(node, dict):
            break
        chain.append(node)
        current = node.get("parent")

    if chain:
        chain.reverse()
        return chain
    return [node for _, node in node_items]


def collect_visible_messages(conversation: dict[str, Any]) -> list[tuple[str, str]]:
    """Return visible messages from the current conversation branch."""
    result: list[tuple[str, str]] = []
    for node in ordered_mapping_nodes(conversation):
        message = node.get("message")
        if isinstance(message, dict):
            result.extend(extract_message_texts(message))
    return result


def sanitize_title(raw_title: Any, output_dir: Path, index: int) -> str:
    """Create a Windows-safe title with a bounded total path length."""
    title = to_single_line(str(raw_title or ""))
    title = INVALID_WINDOWS_FILENAME.sub("_", title)
    title = re.sub(r"\s+", " ", title).strip(" .")

    prefix = f"{index:04d}_"
    suffix = ".txt"
    available = (
        SAFE_WINDOWS_PATH_LENGTH
        - len(str(output_dir.resolve()))
        - 1
        - len(prefix)
        - len(suffix)
    )
    title_limit = min(MAX_TITLE_CHARS, max(1, available))
    title = title[:title_limit].rstrip(" .")
    return title or "未命名对话"


def ensure_directory(path: Path) -> None:
    try:
        filesystem_path(path).mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ConverterError(f"无法创建输出目录：{path}（{exc.strerror or exc}）") from None
    if not filesystem_path(path).is_dir():
        raise ConverterError(f"输出路径不是目录：{path}")


@contextmanager
def atomic_text_writer(path: Path) -> Iterator[TextIO]:
    """Write to a temporary file, replacing the target only after success."""
    ensure_directory(path.parent)
    temp_path: Path | None = None
    try:
        fd, temp_name = tempfile.mkstemp(
            prefix=".chatconv-",
            suffix=".tmp",
            dir=filesystem_path(path.parent),
        )
        temp_path = Path(temp_name)
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as output:
            yield output
        os.replace(filesystem_path(temp_path), filesystem_path(path))
        temp_path = None
    finally:
        if temp_path is not None:
            try:
                filesystem_path(temp_path).unlink(missing_ok=True)
            except OSError:
                pass


def load_conversations(path: Path) -> list[dict[str, Any]]:
    source_path = filesystem_path(path)
    if not source_path.exists():
        raise ConverterError(f"输入文件不存在：{path}")
    if not source_path.is_file():
        raise ConverterError(f"输入路径不是文件：{path}")

    try:
        with source_path.open("r", encoding="utf-8-sig") as source:
            data = json.load(source)
    except MemoryError:
        raise ConverterError(
            "内存不足：输入文件过大。请关闭其他程序，或先拆分导出文件。"
        ) from None
    except UnicodeDecodeError as exc:
        raise ConverterError(f"文件编码错误：不是有效的 UTF-8（位置 {exc.start}）") from None
    except json.JSONDecodeError as exc:
        raise ConverterError(
            f"JSON 格式错误：第 {exc.lineno} 行，第 {exc.colno} 列"
        ) from None
    except OSError as exc:
        raise ConverterError(f"无法读取输入文件：{path}（{exc.strerror or exc}）") from None

    if isinstance(data, dict):
        data = data.get("conversations")
    if not isinstance(data, list):
        raise ConverterError("JSON 结构错误：顶层应为数组，或包含 conversations 数组")

    conversations: list[dict[str, Any]] = []
    for index, item in enumerate(data, 1):
        if not isinstance(item, dict):
            raise ConverterError(f"JSON 结构错误：第 {index} 个对话不是对象")
        conversations.append(item)
    return conversations


def write_message(output: TextIO, speaker: str, content: str) -> None:
    output.write(f"{speaker}：\n{content}\n\n")


def write_single_file(
    conversations: list[dict[str, Any]], output_path: Path
) -> tuple[int, int]:
    ensure_directory(output_path.parent)
    fs_output_path = filesystem_path(output_path)
    if fs_output_path.exists() and fs_output_path.is_dir():
        raise ConverterError(f"输出路径是目录，而不是文件：{output_path}")

    total = len(conversations)
    message_count = 0
    written_dialogues = 0
    skipped_dialogues = 0
    try:
        with atomic_text_writer(output_path) as output:
            for index, conversation in enumerate(conversations, 1):
                messages = collect_visible_messages(conversation)
                if not messages:
                    skipped_dialogues += 1
                    continue

                title = to_single_line(str(conversation.get("title") or "")) or "未命名对话"
                if written_dialogues:
                    output.write("\n")
                output.write(f"【对话 {index}/{total}：{title}】\n\n")
                for speaker, content in messages:
                    write_message(output, speaker, content)
                    message_count += 1
                written_dialogues += 1

            if message_count == 0:
                raise ConverterError(
                    "没有从 JSON 中提取到任何消息。请确认这是 DeepSeek 导出的 conversations.json。"
                )
    except OSError as exc:
        raise ConverterError(f"无法写入输出文件：{output_path}（{exc.strerror or exc}）") from None

    print(
        f"成功写出：1 个文件；对话：{written_dialogues} 个；"
        f"消息：{message_count} 条；跳过空对话：{skipped_dialogues} 个。"
    )
    return written_dialogues, message_count


def write_split_files(
    conversations: list[dict[str, Any]], output_dir: Path
) -> tuple[int, int, int]:
    ensure_directory(output_dir)
    written = failed = skipped = 0

    for index, conversation in enumerate(conversations, 1):
        messages = collect_visible_messages(conversation)
        if not messages:
            skipped += 1
            continue

        title = sanitize_title(conversation.get("title"), output_dir, index)
        output_path = output_dir / f"{index:04d}_{title}.txt"
        try:
            with atomic_text_writer(output_path) as output:
                output.write(f"【对话：{title}】\n\n")
                for speaker, content in messages:
                    write_message(output, speaker, content)
            written += 1
        except OSError as exc:
            failed += 1
            print(
                f"错误：写入失败：{output_path}（{exc.strerror or exc}）",
                file=sys.stderr,
            )

    if written == 0 and failed == 0:
        raise ConverterError(
            "没有从 JSON 中提取到任何消息。请确认这是 DeepSeek 导出的 conversations.json。"
        )

    print(f"成功写出：{written} 个文件；失败：{failed} 个；跳过空对话：{skipped} 个。")
    return written, failed, skipped


def print_stats(conversations: list[dict[str, Any]]) -> None:
    message_count = 0
    character_counts: list[int] = []

    for conversation in conversations:
        for _, content in collect_visible_messages(conversation):
            message_count += 1
            character_counts.append(len(content))

    if character_counts:
        maximum = max(character_counts)
        minimum = min(character_counts)
        median = statistics.median(character_counts)
        median_text = str(int(median)) if float(median).is_integer() else f"{median:.1f}"
    else:
        maximum = minimum = 0
        median_text = "0"

    print(f"总对话数：{len(conversations)}")
    print(f"总消息数：{message_count}")
    print(f"字符数分布：最大 {maximum}，中位数 {median_text}，最小 {minimum}")


def build_parser() -> argparse.ArgumentParser:
    current_dir = Path.cwd()
    parser = argparse.ArgumentParser(
        description="将 DeepSeek conversations.json 转换为可读的 UTF-8 文本"
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {VERSION}",
        help="显示版本号并退出",
    )
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=current_dir / "conversations.json",
        help="输入 JSON 文件（默认：当前目录 conversations.json）",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=current_dir / "chat_clean.txt",
        help="单文件输出路径（默认：当前目录 chat_clean.txt）",
    )
    parser.add_argument(
        "--split-by-conversation",
        action="store_true",
        help="每个对话分别输出为一个 txt 文件",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=current_dir / "chat_split",
        help="分文件模式的输出目录（默认：当前目录 chat_split）",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="只统计对话数、消息数和字符数分布，不写文件",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    conversations = load_conversations(args.input)

    if args.stats:
        print_stats(conversations)
        return 0

    if args.split_by_conversation:
        _, failed, _ = write_split_files(conversations, args.output_dir)
        return 1 if failed else 0

    if args.input.resolve() == args.output.resolve():
        raise ConverterError("输入文件和输出文件不能是同一个路径。")

    write_single_file(conversations, args.output)
    return 0


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return run(args)
    except ConverterError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 1
    except MemoryError:
        print("错误：内存不足，输入文件可能过大。", file=sys.stderr)
        return 1
    except OSError as exc:
        print(f"错误：文件操作失败（{exc.strerror or exc}）", file=sys.stderr)
        return 1
    except Exception as exc:
        print(
            f"错误：处理过程中发生未预期错误（{type(exc).__name__}: {exc}）",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
