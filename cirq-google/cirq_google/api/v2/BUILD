package(default_visibility = ["//visibility:public"])

load("@com_google_protobuf//:protobuf.bzl", "py_proto_library")

proto_library(
    name = "batch_proto",
    srcs = ["batch.proto"],
    deps = [
        ":program_proto",
        ":result_proto",
        ":run_context_proto"
    ],
)

py_proto_library(
    name = "batch_py_proto",
    srcs = ["batch.proto"],
    deps = [
        ":program_py_proto",
        ":result_py_proto",
        ":run_context_py_proto"
    ],
)

cc_proto_library(
    name = "batch_cc_proto",
    deps = [
        ":batch_proto"
    ],
)


proto_library(
    name = "calibration_proto",
    srcs = ["calibration.proto"],
    deps = [
        ":metrics_proto",
        ":program_proto"
    ],
)

py_proto_library(
    name = "calibration_py_proto",
    srcs = ["calibration.proto"],
    deps = [
        ":metrics_py_proto",
        ":program_py_proto"
    ],
)

cc_proto_library(
    name = "calibration_cc_proto",
    deps = [
        ":calibration_proto"
    ],
)

proto_library(
    name = "device_proto",
    srcs = ["device.proto"],
)

py_proto_library(
    name = "device_py_proto",
    srcs = ["device.proto"],
)

cc_proto_library(
    name = "device_cc_proto",
    deps = [
        ":device_proto"
    ],
)

proto_library(
    name = "metrics_proto",
    srcs = ["metrics.proto"],
)

py_proto_library(
    name = "metrics_py_proto",
    srcs = ["metrics.proto"],
)

cc_proto_library(
    name = "metrics_cc_proto",
    deps = [
        ":metrics_proto"
    ],
)

proto_library(
    name = "program_proto",
    srcs = ["program.proto"],
)

py_proto_library(
    name = "program_py_proto",
    srcs = ["program.proto"],
)

cc_proto_library(
    name = "program_cc_proto",
    deps = [
        ":program_proto"
    ],
)

proto_library(
    name = "result_proto",
    srcs = ["result.proto"],
    deps = [
        ":program_proto",
    ],
)

py_proto_library(
    name = "result_py_proto",
    srcs = ["result.proto"],
    deps = [
        ":program_py_proto",
    ],
)

cc_proto_library(
    name = "result_cc_proto",
    deps = [
        ":result_proto",
    ],
)

proto_library(
    name = "run_context_proto",
    srcs = ["run_context.proto"],
    deps = [
        ":program_proto",
    ],
)

py_proto_library(
    name = "run_context_py_proto",
    srcs = ["run_context.proto"],
    deps = [
        ":program_py_proto",
    ],
)

cc_proto_library(
    name = "run_context_cc_proto",
    deps = [
        ":run_context_proto",
    ],
)
