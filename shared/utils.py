def print_args(
    args: object,
) -> None:
    n_char_key = 0
    for key in vars(args):
        n_char_key = max(n_char_key, len(key))
    for key, value in vars(args).items():
        print(f"{key:<{n_char_key + 2}}{value}")
