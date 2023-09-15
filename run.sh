#/bin/bash
# for size in 64 256 512 1024
# for size in 64 256 512 1024 1536 2048 2560 3072 3584 4096
n=$1
for size in 16 32 64 256 512 1000 1001 1024 1536 2048 2560 3072 3584 4096
do
        echo "----- benchmark size: ${size} -----"
        ./build/matMul -k ${n} -s "${size} ${size} ${size}"
done

