import logging
from abc import ABC

class CustomLogger:
    def __init__(self, name, containing_class_name):
        self.logger = logging.getLogger(name)
        self._containing_class_name = containing_class_name
        # Define custom prefixes and suffixes for each log level
        self.level_formats = {
            logging.DEBUG: {"prefix": f"[🔧{self._containing_class_name}] ", "suffix": ""},
            logging.INFO: {"prefix": f"[ℹ️{self._containing_class_name}] ", "suffix": ""},
            logging.WARNING: {"prefix": f"[⚠️{self._containing_class_name}] ", "suffix": ""},
            logging.ERROR: {"prefix": f"[❌{self._containing_class_name}] ", "suffix": ""},
            logging.CRITICAL: {"prefix": f"[‼️{self._containing_class_name}] ", "suffix": ""}
        }

    def _format_message(self, level, msg):
        # Get the prefix and suffix for the log level
        fmt = self.level_formats.get(level, {"prefix": f"[📜{self._containing_class_name}] ", "suffix": ""})
        return f"{fmt['prefix']}{msg}{fmt['suffix']}"

    def debug(self, msg, *args, **kwargs):
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(self._format_message(logging.DEBUG, msg), *args, **kwargs)

    def info(self, msg, *args, **kwargs):
        if self.logger.isEnabledFor(logging.INFO):
            self.logger.info(self._format_message(logging.INFO, msg), *args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if self.logger.isEnabledFor(logging.WARNING):
            self.logger.warning(self._format_message(logging.WARNING, msg), *args, **kwargs)

    def error(self, msg, *args, **kwargs):
        if self.logger.isEnabledFor(logging.ERROR):
            self.logger.error(self._format_message(logging.ERROR, msg), *args, **kwargs)

    def critical(self, msg, *args, **kwargs):
        if self.logger.isEnabledFor(logging.CRITICAL):
            self.logger.critical(self._format_message(logging.CRITICAL, msg), *args, **kwargs)

class HasLoggerClass(ABC):
    logger: CustomLogger

    def __init__(self):
        self.logger = CustomLogger("Orbit." + self.__class__.__name__, self.__class__.__name__)
        self.logger.logger.propagate = True