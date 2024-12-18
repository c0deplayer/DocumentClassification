from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import aiofiles
from Crypto.Cipher import AES
from Crypto.Random import get_random_bytes
from Crypto.Util.Padding import pad, unpad

if TYPE_CHECKING:
    from os import PathLike


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
    ],
)

logger = logging.getLogger(__name__)


class AESCipher:
    """AES Cipher implementation for file encryption and decryption.

    This class provides methods for encrypting and decrypting files using AES in CBC mode.
    It handles large files by processing them in chunks to avoid memory issues.

    Attributes:
        key (bytes): The encryption/decryption key.
        chunk_size (int): Size of chunks to process at a time (default 64KB).

    Example:
        >>> cipher = AESCipher(key=b'your-32-byte-key')
        >>> await cipher.encrypt_file('input.txt', 'encrypted.bin')
        >>> await cipher.decrypt_file('encrypted.bin', 'decrypted.txt')

    Note:
        - The key must be a valid AES key size (16, 24, or 32 bytes)
        - Files are processed asynchronously using aiofiles
        - Uses PKCS7 padding for data that's not a multiple of block size
        - IV (Initialization Vector) is automatically generated and prepended to encrypted file

    """

    def __init__(self, key: bytes) -> None:
        """Initialize OCR class with encryption key and chunk size.

        Args:
            key (bytes): Encryption key used for securing data.

        Attributes:
            key (bytes): Stored encryption key.
            chunk_size (int): Size of data chunks for processing, set to 64KB.

        """
        self.key = key
        self.chunk_size = 64 * 1024  # 64KB chunks

    async def encrypt_file(
        self,
        input_file: PathLike,
        output_file: PathLike,
    ) -> None:
        """Encrypts a file using AES encryption in CBC mode.

        The function reads the input file in chunks, encrypts each chunk using AES-CBC,
        and writes the encrypted data to the output file. The initialization vector (IV)
        is written at the beginning of the output file.

        Args:
            input_file (str): Path to the file to be encrypted
            output_file (str): Path where the encrypted file will be saved

        Returns:
            None

        Raises:
            IOError: If there are issues reading input_file or writing to output_file
            ValueError: If there are encryption related errors

        Note:
            - Uses PKCS7 padding when the final chunk is not a multiple of AES block size
            - Processes file in chunks to handle large files efficiently
            - IV is randomly generated and prepended to encrypted file

        """
        """Encrypt file in chunks to handle large files."""
        iv = get_random_bytes(AES.block_size)
        cipher = AES.new(self.key, AES.MODE_CBC, iv)

        async with aiofiles.open(output_file, "wb") as out_f:
            # Write IV first
            await out_f.write(iv)

            async with aiofiles.open(input_file, "rb") as in_f:
                while chunk := await in_f.read(self.chunk_size):
                    if len(chunk) % AES.block_size != 0:
                        chunk = pad(chunk, AES.block_size)

                    encrypted_chunk = cipher.encrypt(chunk)
                    await out_f.write(encrypted_chunk)

    async def decrypt_file(
        self,
        input_file: PathLike,
        output_file: PathLike | None = None,
    ) -> None:
        """Decrypt file in chunks to handle large files.

        Decrypts a file encrypted using AES-CBC mode, processing in chunks for memory
        efficiency.

        Args:
            input_file: Path to the encrypted file
            output_file: Path for decrypted output (defaults to new temp file)

        Raises:
            ValueError: If file is corrupted or key incorrect
            IOError: If file read/write errors occur

        """
        if output_file is None:
            output_file = input_file.with_suffix(
                input_file.suffix + ".decrypted",
            )

        async with aiofiles.open(input_file, "rb") as in_f:
            # Read IV first
            iv = await in_f.read(AES.block_size)
            cipher = AES.new(self.key, AES.MODE_CBC, iv)

            # Read entire file to handle padding properly
            ciphertext = await in_f.read()
            decrypted = cipher.decrypt(ciphertext)

            try:
                decrypted = unpad(decrypted, AES.block_size)
            except ValueError as e:
                err_msg = "Invalid padding"
                logger.exception("Decryption failed - %s", err_msg)
                raise ValueError(err_msg) from e

            async with aiofiles.open(output_file, "wb") as out_f:
                await out_f.write(decrypted)
