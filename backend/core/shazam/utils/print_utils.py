try:
    from termcolor import colored
    HAS_TERMCOLOR = True
except ImportError:
    HAS_TERMCOLOR = False


def print_error(message):
    """
    打印错误信息
    :param message: 错误提示信息
    :return: None
    """
    if HAS_TERMCOLOR:
        print(colored(message, 'red'))
    else:
        print(f"[ERROR] {message}")


def print_message(message):
    """
    打印正常信息
    :param message: 正常提示信息
    :return: None
    """
    if HAS_TERMCOLOR:
        print(colored(message, 'cyan'))
    else:
        print(f"[INFO] {message}")


def print_warning(message):
    """
    打印警告信息
    :param message: 警告提示信息
    :return: None
    """
    if HAS_TERMCOLOR:
        print(colored(message, 'yellow'))
    else:
        print(f"[WARNING] {message}")
