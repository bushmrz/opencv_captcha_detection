import re


def clear_str(str):
    str = str.lower()
    str = re.sub(
        r"[0123456789\!\#\$\%\^\&\*\(\)\_\~@\`\n\/\|\,\"\<\°\„\?\.\«\’\‚\”\“\®\¥\>\`\'\—\™\‘\:\ \]\[\{\}\=\+\-\\]",
        " ",
        str,
    )
    return str
