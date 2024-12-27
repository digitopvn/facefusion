// ecosystem.config.js

// Load environment variables from .env file
require("dotenv").config();

module.exports = {
	apps: [
		{
			name: process.env.APP_NAME || "facefusion",
			script: "facefusion.py",
			args: ["api"],
			autorestart: true,
			interpreter: "python", // Path to your Python interpreter
			// interpreter:
			// 	"/mnt/digitop_18TB/please-no-not-delete/miniconda3/envs/py310/bin/python",
		},
	],
};
