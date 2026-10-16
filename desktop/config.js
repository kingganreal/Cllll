// Where the web app lives, and which sites may talk to this app's local server.
// Run from the project folder (start.bat), the web app is the copy next to it, served on this PC;
// an installed app opens the published site.
const fs = require('fs');
const path = require('path');

const LOCAL = fs.existsSync(path.join(__dirname, '..', 'index.html'));

module.exports = {
  WEB_APP_URL: LOCAL ? 'http://localhost:8080/' : 'https://kingganreal.github.io/Cllll/',
  ALLOWED_ORIGINS: ['https://kingganreal.github.io'],
  PORT: 8790,
};
