FROM rust:1.96-bullseye AS build
WORKDIR /src
COPY Cargo.toml /src/Cargo.toml
COPY crates /src/crates
COPY deploy/aws/wholetrack /src/deploy/aws/wholetrack
WORKDIR /src/deploy/aws/wholetrack
RUN cargo build --release --locked

FROM public.ecr.aws/lambda/provided:al2023
COPY --from=build /src/deploy/aws/wholetrack/target/release/bootstrap ${LAMBDA_RUNTIME_DIR}/bootstrap
CMD ["bootstrap"]
