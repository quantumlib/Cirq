. $(dirname $0)/container_config.env


USER_EDITOR="${EDITOR:-nano}"
USER_CONTAINERNAME=$(echo ${USER}_${APP_NAME}_${APP_TAG} | sed -e 's#/#_#g;')
USER_APP_ENGINE=${APP_ENGINE:-docker}

function is_running {
    _regex_=$1

    ${USER_APP_ENGINE} ps -a --format "table {{.ID}}:{{.Names}}" | egrep ":${_regex_}$" > /dev/null 2>&1
    rc=$?

    if [ $rc -eq 1 ]; then
        return 1
    fi

    return 0
}

function start_container {
    # -i  = interactive
    # -t  = terminal
    # -rm = cleanup the container after run
    # -v  = map volume
    # -e  = environment variable
    # --name = specify container name

    ${USER_APP_ENGINE} run \
      --name $USER_CONTAINERNAME \
      -e "EDITOR=$USER_EDITOR" \
      -e "HISTFILE=/data/.container_${USER_CONTAINERNAME}_bash_history" \
      -v $HOME/cirq/data:/data -w /data \
      --rm -it $APP_NAME:$APP_TAG /bin/bash
}

function attach_container {
    ${USER_APP_ENGINE} exec -it $USER_CONTAINERNAME /bin/bash
}

if is_running $USER_CONTAINERNAME; then
    echo "# INFO: Attaching to $USER_CONTAINERNAME container already running"
    attach_container
else
    echo "# INFO: Starting $USER_CONTAINERNAME container"
    start_container
fi
