package(default_visibility = ["//visibility:public"])

load("@rules_proto//proto:defs.bzl", "proto_library")
load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

proto_library(
    name = "operations_proto",
    srcs = ["operations.proto"],
)

py_proto_library(
    name = "operations_py_proto",
    deps = [
         ":operations"
    ],
)

cc_proto_library(
    name = "operations_cc_proto",
    srcs = ["operations.proto"],
)

proto_library(
    name = "params_proto",
    srcs = ["params.proto"],
)

proto_library(
    name = "program_proto",
    srcs = ["program.proto"],
    deps = [
        ":operations_proto",
        ":params_proto",
    ],
)
