# Use a multi-stage build
FROM golang:1.24-bookworm AS builder

# Install the lpsolve package
RUN apt-get update && apt-get install -y liblpsolve55-dev

WORKDIR /app
COPY . .

# Set CGO flags for lpsolve package
ENV CGO_CFLAGS="-I/usr/include/lpsolve"
ENV CGO_LDFLAGS="-llpsolve55 -lm -ldl -lcolamd"

# Build all main.go files in cmd directory
RUN for file in $(find cmd -name "main.go"); do \
    dir=$(dirname "$file"); \
    name=$(basename "$dir"); \
    go build -o bin/$name $file; \
  done

# Create the final image
FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y liblpsolve55-dev
COPY --from=builder /app/bin /bin

# Expose the port the API will listen on
EXPOSE 8080

# Command to run the binary when the container starts
CMD ["optimizer"]