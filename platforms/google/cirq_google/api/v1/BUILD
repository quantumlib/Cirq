package(default_visibility = ["//visibility:public"])

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

proto_library(
    name = "operations_proto",
    srcs = ["operations.proto"],
)

py_proto_library(
    name = "operations_py_proto",
    srcs = ["operations.proto"],
)

cc_proto_library(
    name = "operations_cc_proto",
    deps = [
        ":operations_proto",
    ],
)

proto_library(
    name = "params_proto",
    srcs = ["params.proto"],
)

py_proto_library(
    name = "params_py_proto",
    srcs = ["params.proto"],
)

cc_proto_library(
    name = "params_cc_proto",
    deps = [
        ":params_proto",
    ],
)

proto_library(
    name = "program_proto",
    srcs = ["program.proto"],
    deps = [
        ":operations_proto",
        ":params_proto",
    ],
)

py_proto_library(
    name = "program_py_proto",
    srcs = ["program.proto"],
    deps = [
        ":operations_py_proto",
        ":params_py_proto",
    ],
)

cc_proto_library(
    name = "program_cc_proto",
    deps = [
        ":program_proto",
    ],
)
