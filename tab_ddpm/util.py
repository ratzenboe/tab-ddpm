
def get_argparse_defaults(parser):
    defaults = {}
    for action in parser._actions:
        if action.dest != "help" and hasattr(action, "default"):
            defaults[action.dest] = action.default
    return defaults

def try_argparse(parser, *args, **kwargs):
    if __name__ == '__main__':
        return parser.parse_args(*args, **kwargs)
    assert get_argparse_defaults(parser)
