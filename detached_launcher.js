const { spawn } = require('child_process');
const path = require('path');
const fs = require('fs');

const workspaceRoot = __dirname;
const backendDir = path.join(workspaceRoot, 'backend');
const frontendDir = path.join(workspaceRoot, 'frontend');

console.log('==================================================');
console.log('Windows HSL AutoML Server Launcher (v2)');
console.log('==================================================\n');

// Clear existing log files
fs.writeFileSync(path.join(workspaceRoot, 'backend.log'), '');
fs.writeFileSync(path.join(workspaceRoot, 'frontend.log'), '');

const backendLog = fs.openSync(path.join(workspaceRoot, 'backend.log'), 'a');
const frontendLog = fs.openSync(path.join(workspaceRoot, 'frontend.log'), 'a');

// 1. Launch Backend FastAPI
const pythonExe = path.join(backendDir, 'venv', 'Scripts', 'python.exe');
const backendScript = path.join(backendDir, 'run.py');

console.log(`Resolving Python path: ${pythonExe}`);
if (!fs.existsSync(pythonExe)) {
    console.error(`ERROR: Python executable not found at ${pythonExe}`);
    process.exit(1);
}

const backendProcess = spawn(pythonExe, [backendScript], {
    cwd: backendDir,
    detached: true,
    shell: true, // Crucial on Windows to resolve paths properly
    stdio: ['ignore', backendLog, backendLog]
});

backendProcess.on('error', (err) => {
    fs.writeSync(backendLog, `SPAWN ERROR: ${err.message}\n`);
    console.error(`Backend spawn failed: ${err.message}`);
});

backendProcess.unref();
console.log(`Backend launched (PID: ${backendProcess.pid}).`);

// 2. Launch Frontend Vite
console.log('\nResolving Vite node bindings...');
const viteBin = path.join(frontendDir, 'node_modules', 'vite', 'bin', 'vite.js');
if (!fs.existsSync(viteBin)) {
    console.error(`ERROR: Vite bin not found at ${viteBin}`);
    process.exit(1);
}

const frontendProcess = spawn('node', [viteBin], {
    cwd: frontendDir,
    detached: true,
    shell: true, // Crucial on Windows
    stdio: ['ignore', frontendLog, frontendLog]
});

frontendProcess.on('error', (err) => {
    fs.writeSync(frontendLog, `SPAWN ERROR: ${err.message}\n`);
    console.error(`Frontend spawn failed: ${err.message}`);
});

frontendProcess.unref();
console.log(`Frontend launched (PID: ${frontendProcess.pid}).`);

console.log('\n==================================================');
console.log('Spawn triggers completed. Checking logs...');
console.log('==================================================');
process.exit(0);
