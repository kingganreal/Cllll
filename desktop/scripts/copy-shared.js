// The overlay reuses the web app's transparent-video player; copy it in before running or packaging.
const fs = require('fs');
const path = require('path');
const root = path.join(__dirname, '..', '..');
fs.copyFileSync(path.join(root, 'js', 'player.js'), path.join(__dirname, '..', 'renderer', 'player.js'));
fs.copyFileSync(path.join(root, 'icons', 'icon-512.png'), path.join(__dirname, '..', 'assets', 'icon.png'));
console.log('copied player.js and icon');
