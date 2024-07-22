#!/bin/bash

list=(
  "localhost:3050 facefucion-pod-0"
  "localhost:3051 facefucion-pod-1"
  "localhost:3052 facefucion-pod-2"
  "localhost:3053 facefucion-pod-3"
  "localhost:3054 facefucion-pod-4"
  "localhost:3055 facefucion-pod-5"
  "localhost:3056 facefucion-pod-6"
  "localhost:3057 facefucion-pod-7"
)

newTextNginx() {
  local newServer=$1
  echo "upstream faceswap_zii_vn {
  zone upstreams;

  $newServer

  keepalive 16;
}
#End"
}

runCmd() {
  local command=$1
  eval $command
  if [ $? -ne 0 ]; then
    echo "Command failed: $command"
    exit 1
  fi
}

wait_for() {
  local timeout=$1
  sleep $timeout
}

for element in "${list[@]}"; do
  IFS=' ' read -r ip name <<< "$element"
  newText=""
  for server in "${list[@]}"; do
    IFS=' ' read -r server_ip server_name <<< "$server"
    if [ "$server_ip" == "$ip" ]; then
      newText+="server $server_ip max_fails=1 fail_timeout=10s;\n"
    else
      newText+="server $server_ip max_fails=1 fail_timeout=10s down;\n"
    fi
  done

  content=$(newTextNginx "$newText")

  echo -e "$content" > /etc/nginx/conf.d/upstream_proxy.conf
  if [ $? -ne 0 ]; then
    echo "Failed to write to /etc/nginx/conf.d/upstream_proxy.conf"
    continue
  fi
  echo "Updated Nginx config for $name"

  runCmd "sudo nginx -t && sudo nginx -s reload"
  wait_for 60

  runCmd "pm2 restart $name"
  wait_for 10
done
