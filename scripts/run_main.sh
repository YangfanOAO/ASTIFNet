#!/bin/bash

folder_path="./scripts/main"

max_concurrent=1

counter=0

for script in "$folder_path"/*.sh; do

    if [[ $counter -ge $max_concurrent ]]; then
        wait -n
        counter=$((counter - 1))
    fi

    echo "run：$script"
    bash "$script" &

    counter=$((counter + 1))
done

wait
