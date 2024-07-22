const { execSync } = require("child_process");
const fs = require("fs");

const list = [
	{ ip: "localhost:3050", name: "facefucion-pod-0" },
	{ ip: "localhost:3051", name: "facefucion-pod-1" },
	{ ip: "localhost:3052", name: "facefucion-pod-2" },
	{ ip: "localhost:3053", name: "facefucion-pod-3" },
	{ ip: "localhost:3054", name: "facefucion-pod-4" },
	{ ip: "localhost:3055", name: "facefucion-pod-5" },
	{ ip: "localhost:3056", name: "facefucion-pod-6" },
	{ ip: "localhost:3057", name: "facefucion-pod-7" },
];

const defaultText = `upstream faceswap_zii_vn {
	# ip_hash;
	zone upstreams;
	# server localhost:3050;

	server localhost:3050 weight=1 max_fails=1 fail_timeout=10s;
	server localhost:3051 weight=1 max_fails=1 fail_timeout=10s;

	#server localhost:3050 backup;
	# When multiple servers are defined be sure the Host header is not set to one specific destination server.

	# We recommend setting the 'keepalive' parameter to twice the number of servers listed in the upstream block.
	# The proxy_http_version directive should be set to “1.1” and the “Connection” header field should be cleared.
	# Note also that when you specify a load-balancing algorithm in the upstream block – with the hash, ip_hash, least_conn, least_time, or random directive – the directive must appear above the keepalive directive.
	keepalive 16;
}
#End
`;

const newTextNginx = (newServer) => {
	//
	return `upstream faceswap_zii_vn {
	zone upstreams;

	${newServer}

	keepalive 16;
}
#End
`;
};

const runCmd = (command) => {
	try {
		// Full SSH command
		// const command = `sudo nginx -t && sudo nginx -s reload`;

		// Execute command synchronously
		const stdout = execSync(command, { encoding: "utf-8" });

		// Output the result
		console.log(`stdout: ${stdout}`);
	} catch (error) {
		// Handle errors here
		console.error(`Error: ${error.message}`);
		console.error(`stderr: ${error.stderr}`);
		console.error(`Exit code: ${error.status}`);
	}
};

const wait = async (timeout) => {
	await new Promise((resolve, reject) => {
		setTimeout(resolve, timeout);
	});
};

(async () => {
	try {
		for (let i = 0; i < list.length; i++) {
			const element = list[i];
			const newText = list
				.map(({ ip }, index) => {
					const down = index == i ? "down" : "down";
					return `server ${ip} max_fails=1 fail_timeout=10s ${down};`;
				})
				.join("\n");

			const content = newTextNginx(newText);

			try {
				fs.writeFileSync(
					"/etc/nginx/conf.d/upstream_proxy.conf",
					content
				);
			} catch (error) {
				console.error(`fs error`, error);
			}

			(() => {
				// Full SSH command
				const command = `sudo nginx -t && sudo nginx -s reload`;
				runCmd(command);
			})();

			await wait(60000);

			(() => {
				// Full SSH command
				const command = `pm2 restart ${element.name}`;
				runCmd(command);
			})();

			await wait(10000);

			//
		}
	} catch (error) {
		console.error(`metname error`, error);
	}
	//
})();
