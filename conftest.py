def pytest_addoption(parser):
    parser.addoption(
        "--rigetti-integration",
        action="store_true",
        default=False,
        help="run Rigetti integration tests",
    )
