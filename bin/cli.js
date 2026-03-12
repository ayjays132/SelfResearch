#!/usr/bin/env node

const { spawn } = require('child_process');
const path = require('path');
const os = require('os');

/**
 * SelfResearch OS - NPM CLI Wrapper
 * Spawns the python kernel from the global installation directory.
 */

const projectRoot = path.join(__dirname, '..');
const mainScript = path.join(projectRoot, 'main.py');

// Detect python command
const pythonCmd = os.platform() === 'win32' ? 'python' : 'python3';

console.log(`\x1b[32m[SelfResearch OS]\x1b[0m Initializing Neural Kernel...`);

const child = spawn(pythonCmd, [mainScript], {
  cwd: process.cwd(),
  stdio: 'inherit',
  env: {
    ...process.env,
    PYTHONPATH: projectRoot
  }
});

child.on('error', (err) => {
  console.error(`\x1b[31m[Error]\x1b[0m Failed to start the Neural Kernel: ${err.message}`);
  console.log(`Ensure Python 3.10+ is installed and 'python' is in your PATH.`);
  process.exit(1);
});

child.on('exit', (code) => {
  process.exit(code);
});
