set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

# GPU version: base | warp | warpmem | warpmemfound | atomicshare | test2 | test3  (default: warp)
G_version="${1:-warp}"

cd "$SCRIPT_DIR"
rm -rf build
mkdir -p build
cd build
cmake -DCMAKE_BUILD_TYPE=Release -DBuild_Assign=gpu -DG_version="${G_version}" ..
make

cd "$SCRIPT_DIR"
OUTPUT="results_gpu_${G_version}.csv"
echo "percentage,time,iters,time per iterations,inertia" > "$OUTPUT"

for percentage in 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
    echo "=== [${G_version}] Running with ${percentage} of data ==="
    output=$(./build/example "$percentage")
    line=$(echo "$output" | grep "Time:")
    time_ms=$(echo "$line" | sed -n 's/.*Time: \([0-9.]*\) ms.*/\1/p')
    iters=$(echo "$line" | sed -n 's/.*Iterations: \([0-9]*\).*/\1/p')
    tpi=$(echo "$line" | sed -n 's/.*Time per Iteration: \([0-9.]*\) ms.*/\1/p')
    inertia=$(echo "$line" | sed -n 's/.*Inertia: \([0-9.]*\).*/\1/p')
    echo "${percentage},${time_ms},${iters},${tpi},${inertia}" >> "$OUTPUT"
done

echo "Results saved to $OUTPUT"
