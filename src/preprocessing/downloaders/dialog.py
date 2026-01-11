"""Dialog dataset downloaders (Korean Instructions, Persona Chat)."""

import logging
from typing import Iterator

from datasets import load_dataset

from src.preprocessing.downloaders.base import BaseDownloader, RawSample

logger = logging.getLogger(__name__)


class KoreanInstructionsDownloader(BaseDownloader):
    """Korean Instructions dataset downloader.

    Dataset: heegyu/open-korean-instructions
    Size: 200,000+ instruction-response pairs
    Format: Instruction tuning data (ShareGPT style)
    """

    dataset_name = "korean_instructions"
    hf_path = "heegyu/open-korean-instructions"
    expected_size = 200_000

    def download(self) -> None:
        """Download Korean Instructions from HuggingFace."""
        logger.info(f"Downloading {self.dataset_name}...")
        try:
            self.dataset = load_dataset(
                self.hf_path,
                cache_dir=self.cache_dir,
                trust_remote_code=True,
            )
        except Exception as e:
            logger.warning(f"Failed to download {self.dataset_name}: {e}")
            self.dataset = None
        if self.dataset:
            logger.info(f"Downloaded {self.get_stats()}")

    def iterate(self) -> Iterator[RawSample]:
        """Iterate over instruction samples.

        Handles multiple conversation formats:
        - <usr>/<bot> tag format (open-korean-instructions)
        - conversations list format
        - instruction/output format
        """
        if self.dataset is None:
            logger.warning(f"{self.dataset_name} not available, skipping")
            return

        for split in self.dataset.keys():
            for item in self.dataset[split]:
                source_name = item.get("source", self.dataset_name)

                # Format 1: <usr>/<bot> tag format
                if "text" in item:
                    text = item["text"]
                    if "<usr>" in text and "<bot>" in text:
                        # Parse <usr> ... <bot> ... format
                        parts = text.split("<bot>")
                        if len(parts) >= 2:
                            user_part = parts[0].replace("<usr>", "").strip()
                            bot_part = parts[1].strip()

                            if user_part and bot_part:
                                yield RawSample(
                                    text1=user_part,
                                    text2=bot_part,
                                    label="instruction",
                                    source=source_name,
                                    metadata={"split": split},
                                )
                        continue

                # Format 2: conversations list format
                conversations = item.get("conversations", [])
                if len(conversations) >= 2:
                    user_msg = None
                    assistant_msg = None

                    for conv in conversations:
                        role = conv.get("from", conv.get("role", ""))
                        value = conv.get("value", conv.get("content", ""))

                        if role in ("human", "user") and user_msg is None:
                            user_msg = value
                        elif role in ("gpt", "assistant") and assistant_msg is None:
                            assistant_msg = value

                        if user_msg and assistant_msg:
                            break

                    if user_msg and assistant_msg:
                        yield RawSample(
                            text1=user_msg,
                            text2=assistant_msg,
                            label="instruction",
                            source=source_name,
                            metadata={"split": split},
                        )
                    continue

                # Format 3: Simple instruction-output format
                instruction = item.get("instruction", item.get("input", ""))
                output = item.get("output", item.get("response", ""))

                if instruction and output:
                    yield RawSample(
                        text1=instruction,
                        text2=output,
                        label="instruction",
                        source=source_name,
                        metadata={"split": split},
                    )


class PersonaChatDownloader(BaseDownloader):
    """Korean Persona Chat dataset downloader.

    Dataset: NLPBada/korean-persona-chat-dataset
    Size: 10,328 (chat, persona) pairs
    Format: Personalized dialog
    """

    dataset_name = "persona_chat"
    hf_path = "NLPBada/korean-persona-chat-dataset"
    expected_size = 10_328

    def download(self) -> None:
        """Download Persona Chat from HuggingFace."""
        logger.info(f"Downloading {self.dataset_name}...")
        try:
            self.dataset = load_dataset(self.hf_path, cache_dir=self.cache_dir)
        except Exception as e:
            logger.warning(f"Failed to download {self.dataset_name}: {e}")
            self.dataset = None
        if self.dataset:
            logger.info(f"Downloaded {self.get_stats()}")

    def iterate(self) -> Iterator[RawSample]:
        """Iterate over persona chat samples.

        Extracts dialog turns as query-response pairs.
        Handles session_dialog format (stringified list).
        """
        if self.dataset is None:
            logger.warning(f"{self.dataset_name} not available, skipping")
            return

        import ast

        for split in self.dataset.keys():
            for item in self.dataset[split]:
                # Extract persona and chat
                persona = item.get(
                    "session_persona", item.get("persona", "")
                )
                chat = item.get(
                    "session_dialog", item.get("chat", item.get("dialog", ""))
                )

                # Parse stringified list format: "['turn1', 'turn2', ...]"
                if isinstance(chat, str) and chat.startswith("["):
                    try:
                        chat = ast.literal_eval(chat)
                    except (ValueError, SyntaxError):
                        # Fall back to newline split
                        chat = chat.split("\n")

                # Parse persona if stringified list
                if isinstance(persona, str) and persona.startswith("["):
                    try:
                        persona = ast.literal_eval(persona)
                        persona = " ".join(persona) if isinstance(persona, list) else persona
                    except (ValueError, SyntaxError):
                        pass

                # Process chat turns
                if isinstance(chat, list):
                    # List of turns - pair consecutive turns
                    for i in range(0, len(chat) - 1, 2):
                        user_turn = chat[i].strip() if isinstance(chat[i], str) else ""
                        bot_turn = chat[i + 1].strip() if i + 1 < len(chat) and isinstance(chat[i + 1], str) else ""

                        if user_turn and bot_turn:
                            yield RawSample(
                                text1=user_turn,
                                text2=bot_turn,
                                label="dialog",
                                source=self.dataset_name,
                                metadata={
                                    "split": split,
                                    "persona": persona if isinstance(persona, str) else "",
                                },
                            )
                elif isinstance(chat, str):
                    # Newline-separated turns
                    turns = chat.split("\n")
                    for i in range(0, len(turns) - 1, 2):
                        user_turn = turns[i].strip()
                        bot_turn = turns[i + 1].strip() if i + 1 < len(turns) else ""

                        if user_turn and bot_turn:
                            yield RawSample(
                                text1=user_turn,
                                text2=bot_turn,
                                label="dialog",
                                source=self.dataset_name,
                                metadata={
                                    "split": split,
                                    "persona": persona if isinstance(persona, str) else "",
                                },
                            )
