load("@subpar//:subpar.bzl", "par_binary")
load("@my_deps//:requirements.bzl", "requirement")

par_binary(
    name = 'test_binary',
    srcs = ['dev_tools/binary_test/test_binary.py'],
    deps = [':cirq']
)

py_library(
    name = "cirq",
    srcs = glob(
        [
            "cirq/*.py",
            "cirq/**/*.py",
        ],
        exclude = [
            "**/*test.py",
        ],
    ),
    srcs_version = "PY3",
    deps =
        [
#            requirement("quimb"),
#            requirement("google-api-core[grpc]"),
            requirement("matplotlib"),
            requirement("networkx"),
            requirement("numpy"),
            requirement("pandas"),
            requirement("protobuf"),
            requirement("requests"),
            requirement("scipy"),
            requirement("sortedcontainers"),
            requirement("sympy"),
            requirement("typing_extensions"),
            requirement("tqdm"),
        ],
)