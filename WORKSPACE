# The root bazel build file.

# General bazel utility
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Dependencies of protobuf
http_archive(
    name = "bazel_skylib",
    url = "https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
    sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
)

# proto_library, cc_proto_library, and java_proto_library rules implicitly
# depend on @com_google_protobuf for protoc and proto runtimes.
# This statement defines the @com_google_protobuf repo.
http_archive(
    name = "com_google_protobuf",
    sha256 = "b50be32ea806bdb948c22595ba0742c75dc2f8799865def414cf27ea5706f2b7",
    strip_prefix = "protobuf-3.7.0",
    urls = ["https://github.com/google/protobuf/archive/v3.7.0.zip"],
)
