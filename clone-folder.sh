#!/bin/bash
max=1  # Correct max value here

newTextNginx() {
  local newServer=$1
  local keepalive_value=$(((max+1) * 2))
  echo "upstream faceswap_zii_vn {
  zone upstreams;

$newServer

  keepalive $keepalive_value;
}
#End"
}

runCmd() {
  local command=$1
  local user=$2

  if [ -n "$user" ]; then
    su -c "export PATH=\$PATH:/home/ubuntu/.nvm/versions/node/v20.15.1/bin:/home/ubuntu/miniconda3/condabin && $command" - "$user"
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


# Step 1: Copy folder facefusion-0 to facefusion-1 ... facefusion-10
for i in $(seq 0 $max); do
  rm -rf "facefusion-$i"
  cp -r facefusion-0 "facefusion-$i"
  chown ubuntu:ubuntu -R "facefusion-$i"
done

# Step 2: Modify .env files in each folder
for i in $(seq 0 $max); do
  env_file="facefusion-$i/.env"

  # Check if the .env file exists, create if it doesn't
  if [ ! -f "$env_file" ]; then
    touch "$env_file"
  fi

  # Modify the .env file
  {
    echo "PORT=$((3050 + i))"
    echo "DEBUG=True"
    echo "APP_NAME=facefusion-pod-$i"
    echo "OUTPUT_FOLDER_DIR=/home/ubuntu/dynamic_files/swap-face"
    echo "INTERPRETER=/home/ubuntu/miniconda3/envs/facefusion-$i/bin/python"
  } > "$env_file"


   pm2_file="facefusion-$i/ecosystem.config.js"

  # Check if the .env file exists, create if it doesn't
  if [ ! -f "$pm2_file" ]; then
    touch "$pm2_file"
  fi

  # Modify the .env file
  {
  echo "// ecosystem.config.js"
  echo "  "
  echo "// Load environment variables from .env file"
  echo "require('dotenv').config();"
  echo "  "
  echo "module.exports = {"
  echo "	apps: ["
  echo "		{"
  echo "			name: process.env.APP_NAME || 'facefusion',"
  echo "			script: 'run.py',"
  echo "			args: ["
  echo "				'--api',"
  echo "				'--face-enhancer-blend',"
  echo "				35,"
  echo "				'--execution-thread-count',"
  echo "				1,"
    if [ "$i" -lt 3 ]; then
      echo "        '--execution-providers',"
      echo "        'cuda',"
    else
      echo "        // '--execution-providers',"
      echo "        // 'cuda',"
    fi
  echo "			],"
  echo "			autorestart: true,"
  echo "			// interpreter: '/usr/bin/python3', // Path to your Python interpreter"
  echo "			interpreter: process.env.INTERPRETER,"
  echo "		},"
  echo "	],"
  echo "};"
  } > "$pm2_file"
done

# Step 3: Run pm2 start for each folder
for i in $(seq 0 $max); do
  # echo "pm2 start /home/ubuntu/projects/facefusion-$i/ecosystem.config.js" "ubuntu"
  # runCmd "cd /home/ubuntu/projects/facefusion-$i && conda init && conda activate facefusion-$i && pm2 start" "ubuntu"
  runCmd "cd /home/ubuntu/projects/facefusion-$i && bun i && source /home/ubuntu/miniconda3/etc/profile.d/conda.sh && conda activate facefusion-$i && pm2 start /home/ubuntu/projects/facefusion-$i/ecosystem.config.js" "ubuntu"
  # runCmd "cd /home/ubuntu/projects/facefusion-$i && conda init && conda activate facefusion-$i && pm2 start" "ubuntu"
done


# Step 4: Reset all servers to active
newText=""
for i in $(seq 0 $max); do
  newText+="\tserver localhost:$((3050 + i)) weight=1 max_fails=1 fail_timeout=10s;\n"
done

content=$(newTextNginx "$newText")

echo -e "$content" > /etc/nginx/conf.d/upstream_proxy.conf
if [ $? -ne 0 ]; then
  echo "Failed to write to /etc/nginx/conf.d/upstream_proxy.conf"
else
  echo "Reset Nginx config to all servers active"
  runCmd "sudo nginx -t && sudo nginx -s reload" "root"
fi
