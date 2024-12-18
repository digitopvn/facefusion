// ecosystem.config.js

// Load environment variables from .env file
require("dotenv").config();

module.exports = {
	apps: [
		{
			name: process.env.APP_NAME || "facefusion",
			script: "run.py",
			args: [
				"--api",
				"--face-enhancer-blend",
				100,
				"--face-analyser-order",
				"left-right",
				"--face-mask-blur",
				1.0,
				"--execution-thread-count",
				1,
				"--execution-providers",
				"cuda",
			],
			autorestart: true,
			// interpreter: "/usr/bin/python3", // Path to your Python interpreter
			interpreter: process.env.INTERPRETER,
		},
	],
};
