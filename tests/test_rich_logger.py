from src.utils.logger import error, exception, info, setup_logger, warning


def test_logger():
    setup_logger(level="DEBUG", log_format="text")

    info("This is an info message")
    warning("This is a warning message")
    error("This is an error message")

    try:
        1 / 0
    except ZeroDivisionError:
        exception("This is an exception with traceback")


if __name__ == "__main__":
    test_logger()
