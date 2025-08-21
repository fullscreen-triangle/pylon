# Multi-stage Dockerfile for Pylon coordination infrastructure
# Optimized for production deployment with minimal attack surface

# Build stage
FROM rust:1.75-slim-bullseye AS builder

# Install system dependencies for building
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libpq-dev \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /usr/src/pylon

# Copy workspace configuration
COPY Cargo.toml Cargo.lock ./
COPY clippy.toml ./

# Copy source code
COPY crates/ crates/
COPY pylon-config.toml ./

# Build the application in release mode
RUN cargo build --release --workspace

# Runtime stage
FROM debian:bullseye-slim AS runtime

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    libssl1.1 \
    libpq5 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r pylon \
    && useradd -r -g pylon pylon

# Create necessary directories
RUN mkdir -p /etc/pylon/certs \
    && mkdir -p /var/lib/pylon \
    && mkdir -p /var/log/pylon \
    && chown -R pylon:pylon /etc/pylon /var/lib/pylon /var/log/pylon

# Copy built binaries
COPY --from=builder /usr/src/pylon/target/release/pylon-coordinator /usr/local/bin/
COPY --from=builder /usr/src/pylon/target/release/pylon-cli /usr/local/bin/

# Copy configuration
COPY --from=builder /usr/src/pylon/pylon-config.toml /etc/pylon/

# Switch to non-root user
USER pylon

# Expose ports
EXPOSE 8080 9090 50051

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD /usr/local/bin/pylon-cli health-check || exit 1

# Default command
CMD ["/usr/local/bin/pylon-coordinator", "--config", "/etc/pylon/pylon-config.toml"]

# Labels for metadata
LABEL org.opencontainers.image.title="Pylon Coordination Infrastructure"
LABEL org.opencontainers.image.description="Unified spatio-temporal coordination through precision-by-difference calculations"
LABEL org.opencontainers.image.version="0.1.0"
LABEL org.opencontainers.image.authors="Kundai Farai Sachikonye <kundai.sachikonye@wzw.tum.de>"
LABEL org.opencontainers.image.licenses="MIT"
LABEL org.opencontainers.image.source="https://github.com/organization/pylon"
LABEL org.opencontainers.image.documentation="https://pylon-coordination.org"

# Development stage (for development builds)
FROM rust:1.75-slim-bullseye AS development

# Install development dependencies
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    libpq-dev \
    ca-certificates \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install development tools
RUN cargo install cargo-watch cargo-tarpaulin cargo-audit

# Set working directory
WORKDIR /usr/src/pylon

# Copy workspace configuration
COPY Cargo.toml Cargo.lock ./
COPY clippy.toml ./

# Development user setup
RUN groupadd -r developer && useradd -r -g developer developer
RUN chown -R developer:developer /usr/src/pylon

USER developer

# Expose development ports
EXPOSE 8080 9090 50051 8000

# Development command
CMD ["cargo", "run", "--bin", "pylon-coordinator"]
