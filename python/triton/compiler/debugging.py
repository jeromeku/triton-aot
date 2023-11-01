import functools
import json

from loguru import logger

open("debug.log", "w").close()


def serialize(record):
    subset = {"function_info": record["extra"]}
    return json.dumps(subset)


def formatter(record):
    record["extra"]["serialized"] = serialize(record)
    return "{extra[serialized]}\n"


import sys

logger.add(sys.stdout, format="{extra}")
logger.add(
    "debug.log",
    level="DEBUG",
    format="{extra}",  # formatter,
    backtrace=True,
    diagnose=True,
    serialize=True,
)


def trace_func(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with logger.contextualize(function=func.__name__):
            logger.bind(args=args).debug("args")
            logger.bind(kwargs=kwargs).debug("kwargs")
            result = func(*args, **kwargs)
            logger.debug("function.return_value", str(result))
        return result

    return wrapper


# @trace_func
# def add(a, b):
#     return a + b


# add(1, 2)
