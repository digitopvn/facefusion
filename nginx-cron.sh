# #!/bin/bash

# CONFIG_FILE="/etc/nginx/sites-available/load_balancer"
# HOUR=$(date +%H)


# sed -i 's/server backend1.example.com weight=0;/server backend1.example.com weight=10;/g' $CONFIG_FILE
# sed -i 's/server backend2.example.com weight=10;/server backend2.example.com weight=0;/g' $CONFIG_FILE

# pm2 restart server-2


# if (( HOUR % 2 == 0 )); then
#     sed -i 's/server backend1.example.com weight=0;/server backend1.example.com weight=10;/g' $CONFIG_FILE
#     sed -i 's/server backend2.example.com weight=10;/server backend2.example.com weight=0;/g' $CONFIG_FILE
# else
#     sed -i 's/server backend1.example.com weight=10;/server backend1.example.com weight=0;/g' $CONFIG_FILE
#     sed -i 's/server backend2.example.com weight=0;/server backend2.example.com weight=10;/g' $CONFIG_FILE
# fi

# # Test the Nginx configuration and reload if successful
# if sudo nginx -t; then
#     sudo nginx -s reload
# else
#     echo "Nginx configuration test failed. Not reloading."
# fi
