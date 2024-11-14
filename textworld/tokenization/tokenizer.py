import gzip
import html
import os
import sys
from functools import lru_cache
from typing import Dict, List, Set, Tuple, Optional

import ftfy
import regex as re
import logging

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,  # Set to DEBUG to capture all logs; change as needed
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def default_bpe_path() -> str:
    """
    Returns the default path to the BPE vocabulary file.
    
    Returns:
        str: Absolute path to 'bpe_simple_vocab_16e6.txt.gz' located in the same directory as this script.
    """
    try:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), "bpe_simple_vocab_16e6.txt.gz")
    except Exception as e:
        logger.error(f"Error determining BPE file path: {e}")
        raise


@lru_cache(maxsize=1)
def bytes_to_unicode_mapping() -> Dict[int, str]:
    """
    Generates a mapping from byte values to unicode characters.
    
    This ensures reversible BPE operations without introducing unknowns by avoiding
    whitespace/control characters.

    Returns:
        Dict[int, str]: A dictionary mapping byte integers to unique unicode characters.
    """
    try:
        initial_bytes = (
            list(range(ord("!"), ord("~") + 1)) +  # 33-126
            list(range(ord("¡"), ord("¬") + 1)) +  # 161-172
            list(range(ord("®"), ord("ÿ") + 1))    # 174-255
        )
        byte_map = initial_bytes.copy()
        unicode_map = byte_map.copy()
        n = 0
        for b in range(256):
            if b not in byte_map:
                byte_map.append(b)
                unicode_map.append(256 + n)
                n += 1
        unicode_chars = [chr(code_point) for code_point in unicode_map]
        mapping = dict(zip(byte_map, unicode_chars))
        
        # Debug: Verify essential punctuation mappings
        essential_punctuations = [',', '!', '.']
        for punct in essential_punctuations:
            byte_val = ord(punct)
            if byte_val in mapping:
                logger.debug(f"Mapping byte {byte_val} ('{punct}') to Unicode '{mapping[byte_val]}'")
            else:
                logger.error(f"Byte value for punctuation '{punct}' not found in byte_encoder")
        
        # Debug: Verify space character mapping
        if 32 in mapping:
            logger.debug(f"Mapping byte 32 (' ') to Unicode '{mapping[32]}'")
        else:
            logger.error("Byte value for space ' ' not found in byte_encoder")
        
        return mapping
    except Exception as e:
        logger.error(f"Error creating bytes to unicode mapping: {e}")
        raise


def get_symbol_pairs(word: Tuple[str, ...]) -> Set[Tuple[str, str]]:
    """
    Extracts all adjacent symbol pairs in a given word.
    
    Args:
        word (Tuple[str, ...]): A tuple of symbols representing the word.

    Returns:
        Set[Tuple[str, str]]: A set of adjacent symbol pairs.
    """
    try:
        pairs = set()
        if not word:
            return pairs
        prev_char = word[0]
        for char in word[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    except IndexError:
        logger.warning("Received empty word for pair extraction.")
        return set()


def clean_text_basic(text: str) -> str:
    """
    Applies basic cleaning to the input text using ftfy and unescaping HTML entities.

    Args:
        text (str): The raw input text.

    Returns:
        str: The cleaned text.
    """
    try:
        fixed_text = ftfy.fix_text(text)
        unescaped_text = html.unescape(html.unescape(fixed_text))
        return unescaped_text.strip()
    except Exception as e:
        logger.error(f"Error in basic text cleaning: {e}")
        raise


def clean_text_whitespace(text: str) -> str:
    """
    Cleans the whitespace in the text by collapsing multiple spaces into one.

    Args:
        text (str): The input text with potential irregular whitespace.

    Returns:
        str: The whitespace-normalized text.
    """
    try:
        normalized_text = re.sub(r'\s+', ' ', text)
        return normalized_text.strip()
    except Exception as e:
        logger.error(f"Error in whitespace cleaning: {e}")
        raise


class SimpleTokenizer:
    """
    A simple Byte Pair Encoding (BPE) tokenizer.

    Attributes:
        byte_encoder (Dict[int, str]): Mapping from byte integers to unicode characters.
        byte_decoder (Dict[str, int]): Reverse mapping from unicode characters to byte integers.
        encoder (Dict[str, int]): Mapping from BPE tokens to unique integer IDs.
        decoder (Dict[int, str]): Reverse mapping from integer IDs to BPE tokens.
        bpe_ranks (Dict[Tuple[str, str], int]): Ranking of BPE merges.
        cache (Dict[str, str]): Cache for storing already processed tokens.
        pattern (re.Pattern): Compiled regex pattern for tokenization.
    """

    def __init__(self, bpe_path: Optional[str] = None) -> None:
        """
        Initializes the tokenizer with the specified BPE vocabulary file.

        Args:
            bpe_path (Optional[str]): Path to the BPE vocabulary file. Uses default if None.

        Raises:
            FileNotFoundError: If the BPE file does not exist.
            Exception: For unforeseen errors during initialization.
        """
        try:
            self.bpe_path = bpe_path if bpe_path else default_bpe_path()
            if not os.path.isfile(self.bpe_path):
                raise FileNotFoundError(f"BPE file not found at path: {self.bpe_path}")

            self.byte_encoder = bytes_to_unicode_mapping()
            self.byte_decoder = {v: k for k, v in self.byte_encoder.items()}

            # Validate byte_decoder mappings for essential punctuations
            essential_punctuations = [',', '!', '.']
            for punct in essential_punctuations:
                unicode_char = self.byte_encoder.get(ord(punct), None)
                if unicode_char:
                    mapped_byte = self.byte_decoder.get(unicode_char, None)
                    if mapped_byte == ord(punct):
                        logger.debug(f"Punctuation '{punct}' correctly mapped back to byte {mapped_byte}")
                    else:
                        logger.error(f"Punctuation '{punct}' mapped to incorrect byte {mapped_byte}")
                else:
                    logger.error(f"Punctuation '{punct}' not found in byte_encoder")

            # Validate space character mapping
            space_unicode = self.byte_encoder.get(32, None)
            if space_unicode:
                mapped_byte = self.byte_decoder.get(space_unicode, None)
                if mapped_byte == 32:
                    logger.debug(f"Space character ' ' correctly mapped back to byte {mapped_byte}")
                else:
                    logger.error(f"Space character ' ' mapped to incorrect byte {mapped_byte}")
            else:
                logger.error("Space character ' ' not found in byte_encoder")

            with gzip.open(self.bpe_path, 'rt', encoding='utf-8') as bpe_file:
                merges = bpe_file.read().split('\n')[1:]  # Skip the first line (header)

            # Limit the number of merges based on the original code's range
            merges = merges[:49152 - 256 - 2 + 1]

            self.bpe_ranks = {tuple(merge.split()): rank for rank, merge in enumerate(merges) if merge}
            logger.info(f"Loaded {len(self.bpe_ranks)} BPE merge rules.")

            # Initialize vocabulary
            vocab = list(self.byte_encoder.values())
            vocab += [f"{v}</w>" for v in vocab]

            for merge in self.bpe_ranks.keys():
                vocab.append(''.join(merge))
            
            # Add special tokens
            special_tokens = ['<|startoftext|>', '<|endoftext|>']
            vocab.extend(special_tokens)
            self.encoder = {token: idx for idx, token in enumerate(vocab)}
            self.decoder = {idx: token for token, idx in self.encoder.items()}

            # Initialize cache with special tokens
            self.cache: Dict[str, str] = {token: token for token in special_tokens}

            # Compile regex pattern for tokenization
            self.pattern = re.compile(
                r"""<\|startoftext\|>|
                    <\|endoftext\|>|
                    's|'t|'re|'ve|'m|'ll|'d|
                    [\p{L}]+|
                    [\p{N}]|
                    [^\s\p{L}\p{N}]+""",
                re.IGNORECASE | re.VERBOSE
            )
            logger.info("Tokenizer initialization complete.")
        except Exception as e:
            logger.error(f"Failed to initialize SimpleTokenizer: {e}")
            raise

    def bpe(self, token: str) -> str:
        """
        Applies Byte Pair Encoding to a single token.

        Args:
            token (str): The token to encode.

        Returns:
            str: The BPE-encoded token.
        """
        if token in self.cache:
            return self.cache[token]

        word = tuple(token) + ('</w>',)
        pairs = get_symbol_pairs(word)

        if not pairs:
            return token + '</w>'

        while pairs:
            # Select the highest priority pair
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
            if bigram not in self.bpe_ranks:
                break

            rank = self.bpe_ranks[bigram]
            logger.debug(f"Applying BPE merge: {bigram} with rank {rank}")

            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == first and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1

            word = tuple(new_word)
            pairs = get_symbol_pairs(word)

        # Join the tokens with spaces and cache the result
        encoded_token = ' '.join(word)
        self.cache[token] = encoded_token
        return encoded_token

    def encode(self, text: str) -> List[int]:
        """
        Encodes a text string into a list of BPE token IDs.

        Args:
            text (str): The input text to encode.

        Returns:
            List[int]: A list of token IDs representing the encoded text.
        """
        try:
            cleaned_text = clean_text_whitespace(clean_text_basic(text)).lower()
            tokens = self.pattern.findall(cleaned_text)
            bpe_tokens: List[int] = []

            for token in tokens:
                try:
                    byte_encoded = ''.join([self.byte_encoder[b] for b in token.encode('utf-8')])
                except KeyError as e:
                    logger.warning(f"Byte encoding failed for token '{token}': {e}")
                    continue

                bpe_token_str = self.bpe(byte_encoded)
                token_ids = [self.encoder.get(bt, self.encoder.get('<|endoftext|>')) for bt in bpe_token_str.split(' ')]
                bpe_tokens.extend(token_ids)

            logger.debug(f"Encoded tokens: {bpe_tokens}")
            return bpe_tokens
        except Exception as e:
            logger.error(f"Error during encoding: {e}")
            raise

    def decode(self, token_ids: List[int]) -> str:
        """
        Decodes a list of BPE token IDs back into a string.

        Args:
            token_ids (List[int]): The list of token IDs to decode.

        Returns:
            str: The decoded string.
        """
        try:
            tokens = [self.decoder.get(token_id, '') for token_id in token_ids]
            decoded_text = ''
            
            # Retrieve the Unicode representation for space (byte 32)
            space_unicode = self.byte_encoder.get(32)
            if not space_unicode:
                logger.error("Space character mapping not found in byte_encoder.")
                raise ValueError("Space character mapping missing.")

            for token in tokens:
                if token.startswith('<|') and token.endswith('|>'):
                    # Handle special tokens
                    if token == '<|endoftext|>':
                        decoded_text += space_unicode  # Append mapped space
                    # Handle other special tokens if necessary
                    continue  # Skip adding special tokens to the decoded text
                elif token.endswith('</w>'):
                    # Remove the '</w>' suffix and append the mapped space
                    token = token[:-4]
                    decoded_text += token + space_unicode
                else:
                    # Append the token as is (useful for punctuation and regular tokens)
                    decoded_text += token

            logger.debug(f"Decoded text before byte decoding: '{decoded_text}'")

            # Inspect characters that are being mapped to '?'
            problematic_chars = [c for c in decoded_text if c not in self.byte_decoder]
            if problematic_chars:
                logger.warning(f"Characters not found in byte_decoder and will be replaced with '?': {problematic_chars}")

            # Decode byte-encoded characters to their original form
            byte_decoded = bytearray([
                self.byte_decoder.get(c, ord('?')) for c in decoded_text
            ]).decode('utf-8', errors='replace')

            logger.debug(f"Byte decoded text: '{byte_decoded}'")

            # Normalize whitespace by collapsing multiple spaces into one
            final_text = clean_text_whitespace(byte_decoded)
            logger.debug(f"Final decoded text: '{final_text}'")
            return final_text
        except Exception as e:
            logger.error(f"Error during decoding: {e}")
            raise


def main():
    try:
        tokenizer = SimpleTokenizer()  # Uses default BPE path
        sample_text = "Hello, world! This is a test of the SimpleTokenizer."
        encoded = tokenizer.encode(sample_text)
        print(f"Encoded: {encoded[::2]}")

        decoded = tokenizer.decode(encoded)
        print(f"Decoded: '{decoded}'")
    except Exception as e:
        logger.error(f"An error occurred during tokenization: {e}")


if __name__ == "__main__":
    main()