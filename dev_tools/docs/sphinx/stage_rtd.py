def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')
    {"install.md": "install.md"}


if __name__ == '__main__':
    app.run(main)
