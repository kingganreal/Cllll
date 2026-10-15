// The overlay reuses the web app's transparent-video player; copy it in before running or packaging.
const fs = require('fs');
const path = require('path');
const root = path.join(__dirname, '..', '..');
const desk = path.join(__dirname, '..');
fs.mkdirSync(path.join(desk, 'renderer'), { recursive: true });
fs.mkdirSync(path.join(desk, 'assets'), { recursive: true });
fs.copyFileSync(path.join(root, 'js', 'player.js'), path.join(desk, 'renderer', 'player.js'));
fs.copyFileSync(path.join(root, 'icons', 'icon-512.png'), path.join(desk, 'assets', 'icon.png'));
console.log('copied player.js and icon');
