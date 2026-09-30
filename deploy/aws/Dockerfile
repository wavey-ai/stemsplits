# ARM64 container image: the segment Lambda with the weights baked in.
#
# Build for linux/arm64 (Graviton):
#   docker buildx build --platform linux/arm64 --provenance=false --load \
#     -f deploy/aws/Dockerfile -t stems-lambda .
FROM rust:1.96-bookworm AS build
WORKDIR /src
# The crates inherit version/edition/deps from the workspace root, so it has
# to travel with them.
COPY Cargo.toml /src/Cargo.toml
COPY crates /src/crates
COPY deploy/aws/lambda /src/deploy/aws/lambda
WORKDIR /src/deploy/aws/lambda
RUN cargo build --release

FROM public.ecr.aws/lambda/provided:al2023
COPY --from=build /src/deploy/aws/lambda/target/release/bootstrap ${LAMBDA_RUNTIME_DIR}/bootstrap
# The weight bundle travels in the image so a cold start is one file read.
# The f16 bundle holds the same weights as the f32 bundle in half the bytes.
COPY tools/reference/out/bundle-f16 /opt/bundle
ENV BUNDLE_DIR=/opt/bundle
CMD ["bootstrap"]
