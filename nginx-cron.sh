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
  local user=$2

  if [ -n "$user" ]; then
    su -c "export PATH=\$PATH:/home/ubuntu/.nvm/versions/node/v20.15.1/bin && $command" - "$user"
  else
    eval "$command"
  fi

  if [ $? -ne 0 ]; then
    echo "Command failed: $command"
    exit 1
  fi
}


wait_for() {
  local timeout=$1
  sleep $timeout
}

updateNginxConfig() {
  local ip=$1
  local newText=""
  for server in "${list[@]}"; do
    IFS=' ' read -r server_ip server_name <<< "$server"
    if [ "$server_ip" == "$ip" ]; then
      newText+="\tserver $server_ip max_fails=1 fail_timeout=10s down;\n"
    else
      newText+="\tserver $server_ip max_fails=1 fail_timeout=10s;\n"
    fi
  done

  content=$(newTextNginx "$newText")

  echo -e "$content" > /etc/nginx/conf.d/upstream_proxy.conf
  if [ $? -ne 0 ]; then
    echo "Failed to write to /etc/nginx/conf.d/upstream_proxy.conf"
    return 1
  fi
  echo "Updated Nginx config for $ip"
  return 0
}


for element in "${list[@]}"; do
  IFS=' ' read -r ip name <<< "$element"
  
  if updateNginxConfig "$ip"; then
    runCmd "sudo nginx -t && sudo nginx -s reload"
    wait_for 60

    runCmd "pm2 restart $name" "ubuntu"
    wait_for 10
  fi
done

# Reset all servers to active
newText=""
for server in "${list[@]}"; do
  IFS=' ' read -r server_ip server_name <<< "$server"
  newText+="\tserver $server_ip max_fails=1 fail_timeout=10s;\n"
done

content=$(newTextNginx "$newText")

echo -e "$content" > /etc/nginx/conf.d/upstream_proxy.conf
if [ $? -ne 0 ]; then
  echo "Failed to write to /etc/nginx/conf.d/upstream_proxy.conf"
else
  echo "Reset Nginx config to all servers active"
  runCmd "sudo nginx -t && sudo nginx -s reload"
fi
