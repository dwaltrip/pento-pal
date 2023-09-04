import rich
from rich.console import Console
from rich.style import Style


_CONSOLE = Console()

def rich_print(*args, **kwargs):
    return _rich_print(_CONSOLE, *args, **kwargs)

# -----------------------------------------------------------------------------

def _rich_print(console, *args, **kwargs):
    color = kwargs.pop('color', None)
    if color and ('style' in kwargs):
        raise Exception("Error: `color` and `style` can't be used together")

    style = Style(color=color) if color else None
    return console.print(
        *args,
        style=style,
        **kwargs,
    )

# -----------------------------------------------------------------------------

class Logger:
    def __init__(self):
        self.console = Console()

    def log(self, *args, **kwargs):
        return _rich_print(self.console, *args, **kwargs)

logger = Logger()