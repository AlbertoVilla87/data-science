"""
TODO
"""
import logging

from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class DsLogger(logging.Logger):
    """
    TODO
    """
    _c_style = "%(asctime)s - %(levelname)s - %(message)s"
    _f_style = "%(asctime)s - %(levelname)s - %(module)s - %(funcName)s - %(message)s"

    def __init__(self, name):
        # Formatter
        super().__init__(name)

        # Console handler
        self._c_hand = None

        # File handler
        self._f_hand = None

        # Rotating file handler
        self._rf_hand = None

        # Timed rotating file handler
        self._tf_hand = None

        # Init console handler
        self.set_console_logs(logging.INFO)


    def set_console_logs(self, level):
        """
        TODO
        """
        # Remove old handler
        self.unset_console_logs()

        # New handler
        self._c_hand = logging.StreamHandler()
        self._c_hand.setLevel(level)
        self._c_hand.setFormatter(logging.Formatter(DsLogger._c_style))

        # Add handler to logger
        self.addHandler(self._c_hand)

    def unset_console_logs(self):
        """
        TODO
        """
        # Remove old handler
        if self._c_hand:
            self.removeHandler(self._c_hand)

        self._c_hand = None

    def set_file_logs(self, level, filename="logs/app.log", mode="w"):
        """
        TODO
        """
        self.unset_file_logs()

        # New handler
        self._f_hand = logging.FileHandler(filename, mode=mode)
        self._f_hand.setLevel(level)
        self._f_hand.setFormatter(logging.Formatter(DsLogger._f_style))

        # Add handler to logger
        self.addHandler(self._f_hand)

    def unset_file_logs(self):
        """
        TODO
        """
        # Remove old handler
        if self._f_hand:
            self.removeHandler(self._f_hand)

        self._f_hand = None

    def set_rotating_logs(self,
                          level,
                          filename="logs/app.log",
                          max_bytes=5000000, # ~5MB
                          backup_count=5):
        """
        TODO
        """
        self.unset_rotating_logs()

        # New handler
        self._rf_hand = RotatingFileHandler(filename,
                                            maxBytes=max_bytes,
                                            backupCount=backup_count)
        self._rf_hand.setLevel(level)
        self._rf_hand.setFormatter(logging.Formatter(DsLogger._f_style))

        # Add handler to logger
        self.addHandler(self._rf_hand)

    def unset_rotating_logs(self):
        """
        TODO
        """
        if self._rf_hand:
            self.removeHandler(self._rf_hand)

        self._rf_hand = None

    def set_timed_rotating_logs(self,
                                level,
                                filename="logs/app.log",
                                when="D",
                                interval=1,
                                backupCount=7):
        """
        TODO
        """
        self.unset_timed_rotating_logs()

        # New handler
        self._tf_hand = TimedRotatingFileHandler(filename,
                                                 when=when,
                                                 interval=interval,
                                                 backupCount=backupCount)
        self._tf_hand.setLevel(level)
        self._tf_hand.setFormatter(logging.Formatter(DsLogger._f_style))

        # Add handler to logger
        self.addHandler(self._tf_hand)

    def unset_timed_rotating_logs(self):
        """
        TODO
        """
        if self._tf_hand:
            self.removeHandler(self._tf_hand)

        self._tf_hand = None

    def close(self):
        """
        TODO
        """
        for hanlder in [x for x in (self._f_hand, self._rf_hand, self._tf_hand) if x]:
            hanlder.close()