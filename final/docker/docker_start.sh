
#!/usr/bin/env bash

BASH_OPTION=bash

IMG=iscilab/yolov10
containerid=$(docker ps -qf "ancestor=${IMG}") && echo $containerid

xhost +

if [[ -n "$containerid" ]]
then
    docker exec -it \
        --privileged \
        -e DISPLAY=${DISPLAY} \
        -e LINES="$(tput lines)" \
        yolov10 \
        $BASH_OPTION
else
    docker start -i yolov10
fi
