set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# Update-step version for kmeans_v1.c (1=thread-local, 2=atomic, 3=reduction).
# Usage: ./run.sh [version]  (default: 3)
UPDATE_VERSION="${1:-3}"

cd "$SCRIPT_DIR"
# Clean build dir so the new -DUPDATE_VERSION actually takes effect
rm -rf build
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_TARGET=cpu -DUPDATE_VERSION="${UPDATE_VERSION}" ..
make

cd "$SCRIPT_DIR"
OUTPUT="results_v${UPDATE_VERSION}.csv"
echo "percentage,time,iters,time per iterations,inertia" > "$OUTPUT"

for percentage in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    echo "=== Running with ${percentage} of data ==="
    output=$(./build/example "$percentage")
    line=$(echo "$output" | grep "Time:")
    time_ms=$(echo "$line" | sed -n 's/.*Time: \([0-9.]*\) ms.*/\1/p')
    iters=$(echo "$line" | sed -n 's/.*Iterations: \([0-9]*\).*/\1/p')
    tpi=$(echo "$line" | sed -n 's/.*Time per Iteration: \([0-9.]*\) ms.*/\1/p')
    inertia=$(echo "$line" | sed -n 's/.*Inertia: \([0-9.]*\).*/\1/p')
    echo "${percentage},${time_ms},${iters},${tpi},${inertia}" >> "$OUTPUT"
done

echo "Results saved to $OUTPUT"
