"""Conflog logger configuration for TicTacToe Policy."""

import logging
import sys

from conflog import Conflog


def init(
    name: str,
    *,
    stdout: bool = False,
    message_only: bool = False,
) -> logging.LoggerAdapter:
    """Create a configured logger for a package module.

    :param name: Logger name, normally the calling module's ``__name__``.
    :param stdout: Send records to stdout instead of the default stderr.
    :param message_only: Omit package name and level from the output format.
    :returns: A Conflog-configured logger adapter.
    """
    output_format = (
        "%(message)s"
        if message_only
        else "[tictactoe-policy] %(levelname)s %(message)s"
    )
    conflog = Conflog(
        conf_dict={
            "level": "info",
            "format": output_format,
        }
    )
    logger = conflog.get_logger(name)
    if stdout:
        for handler in conflog.handlers:
            handler.setStream(sys.stdout)
    return logger
