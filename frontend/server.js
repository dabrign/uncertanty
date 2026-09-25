import express from 'express';
import path from 'path';
import { fileURLToPath } from 'url';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

// Serve static assets built by Vite
const staticDir = path.resolve(__dirname, '../website/dashboard_app');
app.use(express.static(staticDir));

app.get('*', (req, res) => {
  res.sendFile(path.join(staticDir, 'index.html'));
});

app.listen(PORT, () => {
  console.log(`======================================================================`);
  console.log(`  React HITL Dashboard Node Server running at http://localhost:${PORT}`);
  console.log(`======================================================================`);
});
