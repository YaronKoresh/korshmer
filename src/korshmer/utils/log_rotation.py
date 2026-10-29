import gzip
import logging.handlers
import os
import shutil

COMPRESSED_SUFFIX = ".gz"


def _gzip_rotator(source, dest):
    tmp = dest + ".tmp"
    try:
        with open(source, "rb") as src, gzip.open(tmp, "wb") as out:
            shutil.copyfileobj(src, out)
        os.replace(tmp, dest)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    os.remove(source)


class CompressingRotatingFileHandler(logging.handlers.RotatingFileHandler):
    def __init__(
        self, filename, maxBytes=0, backupCount=0, encoding=None, delay=False
    ):
        super().__init__(
            filename,
            maxBytes=maxBytes,
            backupCount=backupCount,
            encoding=encoding,
            delay=delay,
        )
        self.namer = lambda name: name + COMPRESSED_SUFFIX
        self.rotator = _gzip_rotator
