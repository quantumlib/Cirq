for folder in cirq/*; do
for filename in $folder/*; do
sed -i 's/numpy/cupy/' $filename
done
done

for folder in cirq/google; do
for filename in $folder/*; do
sed -i 's/numpy/cupy/' $filename
done
done

for folder in cirq/google/line; do
for filename in $folder/*; do
sed -i 's/numpy/cupy/' $filename
done
done

for folder in cirq/google/api; do
for filename in $folder/*; do
sed -i 's/numpy/cupy/' $filename
done
done


for filename in in cirq/*.py; do
sed -i 's/numpy/cupy/' $filename
done